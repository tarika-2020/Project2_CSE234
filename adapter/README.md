Place the final QLoRA adapter files in this directory.

Expected runtime behavior:
- `main.py` loads the public base model from HuggingFace Hub.
- If `adapter/adapter_config.json` exists, it attaches the local PEFT adapter.
- If no adapter is present or the ML stack is unavailable, the repo falls back to a deterministic heuristic baseline.
