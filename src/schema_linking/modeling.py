import os
from dataclasses import dataclass
from typing import List


@dataclass
class GenerationResult:
    text: str
    backend_name: str


class GenerationBackend:
    name = "base"

    def generate(self, prompts: List[str]) -> List[GenerationResult]:
        raise NotImplementedError


class HeuristicBackend(GenerationBackend):
    name = "heuristic"

    def generate(self, prompts: List[str]) -> List[GenerationResult]:
        return [GenerationResult(text="{}", backend_name=self.name) for _ in prompts]


class TransformersBackend(GenerationBackend):
    name = "transformers"

    def __init__(self, base_model: str, adapter_dir: str, max_new_tokens: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            trust_remote_code=True,
        )

        if os.path.isdir(adapter_dir) and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)

        self.model = model.eval()

    def generate(self, prompts: List[str]) -> List[GenerationResult]:
        batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        output = self.model.generate(
            **batch,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = batch["input_ids"].shape[1]
        generations = []
        for row in output:
            new_tokens = row[prompt_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            generations.append(GenerationResult(text=text, backend_name=self.name))
        return generations


def build_generation_backend(base_model: str, adapter_dir: str, max_new_tokens: int) -> GenerationBackend:
    try:
        import importlib.util

        transformers_ok = importlib.util.find_spec("transformers") is not None
        if not transformers_ok:
            return HeuristicBackend()
        adapter_config = os.path.join(adapter_dir, "adapter_config.json")
        if not os.path.exists(adapter_config):
            return HeuristicBackend()
        return TransformersBackend(base_model=base_model, adapter_dir=adapter_dir, max_new_tokens=max_new_tokens)
    except Exception:
        return HeuristicBackend()
