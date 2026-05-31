import argparse
import json
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def require_package(name: str) -> None:
    import importlib.util

    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"Missing required package: {name}")


def ensure_utf8_mode() -> None:
    if sys.flags.utf8_mode:
        return
    if os.environ.get("PROJECT2_UTF8_REEXEC") == "1":
        return
    env = os.environ.copy()
    env["PROJECT2_UTF8_REEXEC"] = "1"
    cmd = [sys.executable, "-X", "utf8", *sys.argv]
    raise SystemExit(subprocess.call(cmd, env=env, cwd=os.getcwd()))


def ensure_local_hf_cache() -> None:
    cache_root = os.path.join(ROOT, ".hf_cache")
    os.makedirs(cache_root, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(cache_root, "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_root, "datasets"))


def main() -> None:
    ensure_utf8_mode()
    ensure_local_hf_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_dir", default="./adapter")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--config_out", default=None)
    args = parser.parse_args()

    for package in ("torch", "transformers", "peft", "trl", "datasets"):
        require_package(package)

    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer
    import torch

    dataset = load_dataset("json", data_files={"train": args.train_file, "eval": args.eval_file})
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if args.load_in_4bit:
        require_package("bitsandbytes")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    use_cpu = not torch.cuda.is_available()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[module.strip() for module in args.target_modules.split(",") if module.strip()],
    )
    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_seq_length,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        save_steps=args.save_steps,
        report_to=[],
        bf16=args.bf16 if not use_cpu else False,
        fp16=False,
        use_cpu=use_cpu,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.config_out:
        with open(args.config_out, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
