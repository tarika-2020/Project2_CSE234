# Raw Output Files

This folder contains copied JSON prediction outputs produced by validation and
replay runs.

## File-to-Run Mapping

- `validation_predictions.json`
  - Run type: local heuristic/fallback validation output
  - Main knobs: no adapter loaded, deterministic fallback path only
  - Purpose: baseline raw JSON for the heuristic pipeline

- `validation_predictions_venv.json`
  - Run type: local `.venv` validation output
  - Main knobs: same fallback-style runtime, different local environment
  - Purpose: environment-specific validation replay

- `venv_validation_predictions.json`
  - Run type: local `.venv` validation output from a later replay
  - Main knobs: same validation input, local environment replay
  - Purpose: repeated local validation artifact kept for debugging

- `final_validation_predictions.json`
  - Run type: final local fallback validation output
  - Main knobs: deterministic heuristic/retrieval path
  - Purpose: final saved raw JSON for the non-adapter baseline

- `final_validation_predictions_best_adapter.json`
  - Run type: best adapter raw predictions before confidence gating
  - Corresponding model knobs:
    - Base model: `Qwen/Qwen2.5-1.5B-Instruct`
    - Schema variant: filtered
    - Train set: original
    - Epochs: 2
    - Learning rate: `1.5e-4`
    - LoRA rank: 16
  - Purpose: primary model output before fallback gating

- `final_validation_predictions_best_adapter_gated.json`
  - Run type: best adapter predictions after confidence gating
  - Corresponding model knobs:
    - Base model: `Qwen/Qwen2.5-1.5B-Instruct`
    - Schema variant: filtered
    - Train set: original
    - Epochs: 2
    - Learning rate: `1.5e-4`
    - LoRA rank: 16
    - Inference change: confidence-gated fallback swapping
  - Purpose: strongest saved validation output
  - Verified score: `0.5511`

- `composed_gated_predictions.json`
  - Run type: reproduced composed version of the gated best output
  - Corresponding knobs:
    - Same primary model as `final_validation_predictions_best_adapter.json`
    - Same fallback source as `final_validation_predictions.json`
    - Same gate logic as the runtime fallback gate
  - Purpose: reproducible reconstruction of the `0.5511` gated output

## Notes

- Not every experiment listed in `logs/experiment_results.csv` has a raw JSON
  output in this folder, because several runs were executed on EC2 and only
  their final scores were retained locally.
- The full scored experiment table is in `logs/experiment_results.csv`.
