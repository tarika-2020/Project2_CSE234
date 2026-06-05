# Experiment Logs

This directory contains both experiment summaries and the actual prediction
outputs collected from runs during development.

Contents:

- `experiment_results.csv`: flat table of the main scored configurations
- `experiment_summary.md`: readable summary of the runs and the selected final path
- `raw_outputs/`: copied JSON prediction outputs from validation and replay runs
- `scored_outputs/`: copied per-question CSV outputs from scored runs

The final selected submission path is:

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter: `ec2_qwen25_1p5b_filtered_e2_lr15e5`
- Inference entrypoint: `main.py`
- Verified validation leaderboard score: `0.5511`
