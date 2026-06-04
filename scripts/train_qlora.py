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
    cache_root = os.environ.get("PROJECT2_CACHE_DIR")
    if not cache_root:
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            cache_root = "/kaggle/working/.hf_cache"
        else:
            cache_root = os.path.join(ROOT, ".hf_cache")
    os.makedirs(cache_root, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(cache_root, "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_root, "datasets"))


def bf16_supported(torch_module) -> bool:
    if not torch_module.cuda.is_available():
        return False
    checker = getattr(torch_module.cuda, "is_bf16_supported", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            pass
    major, _minor = torch_module.cuda.get_device_capability()
    return major >= 8


def should_disable_thinking(base_model: str) -> bool:
    return "qwen3" in base_model.lower()


def render_messages_with_chat_template(tokenizer, messages, disable_thinking: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    if disable_thinking:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


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
    parser.add_argument("--single_gpu_device", type=int, default=0)
    parser.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--config_out", default=None)
    parser.add_argument("--report_to", default="")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--logging_dir", default=None)
    args = parser.parse_args()

    report_to = [item.strip() for item in args.report_to.split(",") if item.strip()]

    required_packages = ["torch", "transformers", "peft", "trl", "datasets"]
    if "mlflow" in report_to:
        required_packages.append("mlflow")
    for package in required_packages:
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
    disable_thinking = should_disable_thinking(args.base_model)

    if disable_thinking and hasattr(tokenizer, "apply_chat_template"):
        dataset = dataset.map(
            lambda row: {
                "text": render_messages_with_chat_template(
                    tokenizer,
                    row["messages"],
                    disable_thinking=True,
                )
            },
            desc="Applying chat template",
        )

    text_only_columns = ["text"]
    for split_name in dataset.keys():
        removable = [
            column_name
            for column_name in dataset[split_name].column_names
            if column_name not in text_only_columns
        ]
        if removable:
            dataset[split_name] = dataset[split_name].remove_columns(removable)

    use_cpu = not torch.cuda.is_available()
    use_bf16 = args.bf16 and bf16_supported(torch)

    model_kwargs = {
        "trust_remote_code": True,
    }
    if use_cpu:
        model_kwargs["device_map"] = "cpu"
    else:
        model_kwargs["device_map"] = {"": args.single_gpu_device}
    if args.load_in_4bit:
        require_package("bitsandbytes")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

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
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_seq_length,
        logging_steps=10,
        logging_dir=args.logging_dir,
        eval_strategy="epoch",
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        save_steps=args.save_steps,
        report_to=report_to,
        bf16=use_bf16 if not use_cpu else False,
        fp16=(not use_bf16) if not use_cpu else False,
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
        config_payload = dict(vars(args))
        config_payload["resolved_bf16"] = use_bf16
        config_payload["resolved_fp16"] = (not use_bf16) if not use_cpu else False
        with open(args.config_out, "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)


if __name__ == "__main__":
    main()
