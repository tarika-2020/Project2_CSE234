# Project2_CSE234

Submission-ready schema-linking pipeline for CSE/DSC 234 Project 2.

## Repo Layout

- `main.py`: TA-facing inference entrypoint
- `schemas/`: release schemas copied to the repo root
- `src/schema_linking/`: shared loading, filtering, prompting, decoding, and fallback logic
- `scripts/prepare_data.py`: builds SFT-ready JSONL datasets
- `scripts/train_qlora.py`: local QLoRA training entrypoint
- `scripts/run_validation.py`: runs validation inference and grading
- `scripts/eval_predictions.py`: wrapper around the provided `eval.py`
- `adapter/`: place the final PEFT adapter here
- `artifacts/`: generated outputs
- `logs/`: experiment logs
- `eval.py`: provided grader copied to repo root for TA-compatible evaluation

## Install

Validated local setup for this repo uses Python `3.11` via:

```bash
C:\Users\Admin\miniconda3\envs\biomni_e1\python.exe -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Then run commands with `.\.venv\Scripts\python`.

Minimal install command:

```bash
pip install -r requirements.txt
```

Recommended packages for the intended training/inference path:

- `torch`
- `transformers`
- `peft`
- `trl`
- `datasets`
- `accelerate`
- `bitsandbytes`
- `sentencepiece`

If these packages are not installed, `main.py` still runs with a deterministic heuristic fallback so the interface remains testable.

Current environment note:

- The validated `.venv` on this machine imports the full stack successfully.
- The installed Torch build is currently CPU-only, so training works in CPU mode here unless a CUDA-enabled Torch build is installed in an environment with visible NVIDIA drivers.

## Runtime Contract

The required inference command is:

```bash
python main.py --input input_filename --output output_filename
```

Optional flags:

```bash
python main.py \
  --input validation_input.json \
  --output artifacts/validation_predictions.json \
  --schemas_dir ./schemas \
  --adapter_dir ./adapter \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --max_new_tokens 256
```

Important runtime behavior:

- Schemas are loaded from `./schemas` by default.
- If `adapter/adapter_config.json` exists and the ML stack is installed, the runtime loads the public base model and attaches the local adapter.
- If the adapter is missing or the ML stack is unavailable, the runtime falls back to a deterministic two-stage heuristic pipeline.
- Output never includes `db_id`.

## Prepare Training Data

Build filtered-schema and full-schema JSONL datasets:

```bash
.\.venv\Scripts\python scripts/prepare_data.py
```

This writes files under `artifacts/prepared/`, including:

- `train_filtered.jsonl`
- `validation_filtered.jsonl`
- `train_full_schema.jsonl`
- `validation_full_schema.jsonl`

## Train QLoRA Adapter

Example command:

```bash
.\.venv\Scripts\python scripts/train_qlora.py \
  --train_file artifacts/prepared/train_filtered.jsonl \
  --eval_file artifacts/prepared/validation_filtered.jsonl \
  --output_dir adapter \
  --load_in_4bit \
  --bf16
```

Notes:

- The default base model is `Qwen/Qwen2.5-1.5B-Instruct`.
- `--load_in_4bit` enables the intended QLoRA path and requires `bitsandbytes`.
- On a CPU-only machine, omit `--bf16` and expect a slow fallback training path.
- The training script redirects `HF_HOME`, hub cache, and datasets cache into repo-local `.hf_cache/` so it does not depend on profile-level write access.
- The script expects the HuggingFace model to be accessible from the environment where training runs.
- If HuggingFace access requires authentication, log in before training or inference.

## Generate Experiment Configs

Write the planned experiment matrix for the report:

```bash
.\.venv\Scripts\python scripts/generate_experiment_configs.py
```

This creates JSON config stubs under `artifacts/experiment_configs/`.

## Run Validation

Generate predictions and score them with the provided grader:

```bash
.\.venv\Scripts\python scripts/run_validation.py
```

This writes:

- `artifacts/validation_predictions.json`
- `artifacts/per_question.csv`

Inspect the hardest failures by question and database:

```bash
.\.venv\Scripts\python scripts/diagnose_predictions.py \
  --predictions artifacts/validation_predictions.json \
  --per_question artifacts/per_question.csv
```

## Packaging Notes

- Keep the final learned PEFT adapter in `adapter/`.
- Keep `schemas/` at the repo root.
- Keep experiment outputs and logs in `logs/`.
- The base model is expected to be pulled from HuggingFace Hub at runtime; the adapter remains local to the repo.

## Current Status

Implemented now:

- reusable schema loader and serializer
- deterministic schema candidate filter
- prompt builder
- JSON extraction and schema-aware cleanup
- TA-facing `main.py`
- SFT data preparation script
- local training script scaffold
- validation/evaluation wrappers

Not included yet:

- trained adapter weights
- final report PDF
- RapidFire experiment logs
