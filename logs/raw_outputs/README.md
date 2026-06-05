# Raw Output Files

This folder contains copied JSON prediction outputs produced by validation and
replay runs.

## File-to-Run Mapping

Scored experiment configs and their raw JSON outputs:

- `C1` -> `validation_predictions_ec2_qwen25_0p5b_filtered_e1.json`
  - Base model: `Qwen/Qwen2.5-0.5B-Instruct`
  - Schema variant: filtered
  - Train set: original
  - Epochs: 1
  - Learning rate: `2e-4`
  - LoRA rank: 16

- `C2` -> `validation_predictions_ec2_qwen25_0p5b_filtered_lr1e4.json`
  - Base model: `Qwen/Qwen2.5-0.5B-Instruct`
  - Schema variant: filtered
  - Train set: original
  - Epochs: 1
  - Learning rate: `1e-4`
  - LoRA rank: 16

- `C3` -> `validation_predictions_ec2_qwen25_1p5b_filtered_e1.json`
  - Base model: `Qwen/Qwen2.5-1.5B-Instruct`
  - Schema variant: filtered
  - Train set: original
  - Epochs: 1
  - Learning rate: `2e-4`
  - LoRA rank: 16

- `C4` -> `final_validation_predictions_best_adapter.json`
  - Base model: `Qwen/Qwen2.5-1.5B-Instruct`
  - Schema variant: filtered
  - Train set: original
  - Epochs: 2
  - Learning rate: `1.5e-4`
  - LoRA rank: 16

- `C5` -> `validation_predictions_ec2_qwen25_1p5b_filtered_e5_lr15e5.json`
  - Base model: `Qwen/Qwen2.5-1.5B-Instruct`
  - Schema variant: filtered
  - Train set: original
  - Epochs: 5
  - Learning rate: `1.5e-4`
  - LoRA rank: 16

- `C6` -> `validation_predictions_ec2_qwen25_1p5b_full_e5_lr15e5.json`
  - Base model: `Qwen/Qwen2.5-1.5B-Instruct`
  - Schema variant: full
  - Train set: original
  - Epochs: 5
  - Learning rate: `1.5e-4`
  - LoRA rank: 16

- `C7` -> `validation_predictions_ec2_qwen25_1p5b_augbalanced_filtered_e2_lr15e5.json`
  - Base model: `Qwen/Qwen2.5-1.5B-Instruct`
  - Schema variant: filtered
  - Train set: augmented-balanced
  - Epochs: 2
  - Learning rate: `1.5e-4`
  - LoRA rank: 16

- `C8` -> `validation_predictions_ec2_qwen25_1p5b_filtered_e2_lr15e5_r32.json`
  - Base model: `Qwen/Qwen2.5-1.5B-Instruct`
  - Schema variant: filtered
  - Train set: original
  - Epochs: 2
  - Learning rate: `1.5e-4`
  - LoRA rank: 32

- `F1` -> `final_validation_predictions_best_adapter_gated.json`
  - Base model: `Qwen/Qwen2.5-1.5B-Instruct`
  - Schema variant: filtered
  - Train set: original
  - Epochs: 2
  - Learning rate: `1.5e-4`
  - LoRA rank: 16
  - Inference change: confidence-gated fallback swapping
  - Verified score: `0.5511`

Other local replay/debug raw outputs in this folder:

- `validation_predictions.json`
  - Local heuristic/fallback validation output

- `validation_predictions_venv.json`
  - Local `.venv` validation replay

- `venv_validation_predictions.json`
  - Later local `.venv` validation replay

- `final_validation_predictions.json`
  - Final local fallback validation output

- `final_validation_predictions_best_adapter.json`
  - Saved best adapter raw predictions before fallback gating

- `composed_gated_predictions.json`
  - Reproduced composed version of the gated best output

## Notes

- The full scored experiment table is in `logs/experiment_results.csv`.
- The main scored config outputs are the `C1` through `C8` and `F1` mappings above.
