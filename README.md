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

Create an environment and install:

```bash
pip install -r requirements.txt
```

Recommended packages for the intended training/inference path:

- `torch`
- `transformers`
- `peft`
- `trl`
- `datasets`

If these packages are not installed, `main.py` still runs with a deterministic heuristic fallback so the interface remains testable.

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
python scripts/prepare_data.py
```

This writes files under `artifacts/prepared/`, including:

- `train_filtered.jsonl`
- `validation_filtered.jsonl`
- `train_full_schema.jsonl`
- `validation_full_schema.jsonl`

## Train QLoRA Adapter

Example command:

```bash
python scripts/train_qlora.py \
  --train_file artifacts/prepared/train_filtered.jsonl \
  --eval_file artifacts/prepared/validation_filtered.jsonl \
  --output_dir adapter
```

Notes:

- The default base model is `Qwen/Qwen2.5-1.5B-Instruct`.
- The script expects the HuggingFace model to be accessible from the environment where training runs.
- If HuggingFace access requires authentication, log in before training or inference.

## Run Validation

Generate predictions and score them with the provided grader:

```bash
python scripts/run_validation.py
```

This writes:

- `artifacts/validation_predictions.json`
- `artifacts/per_question.csv`

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
