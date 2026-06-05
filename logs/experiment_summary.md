# Experiment Summary

This folder records the scored runs used to choose the final submission model.

## Main Scored Configurations

| ID | Base model | Schema | Train set | Epochs | LR | LoRA r | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | Qwen2.5-0.5B | filtered | original | 1 | 2e-4 | 16 | 0.4314 |
| C2 | Qwen2.5-0.5B | filtered | original | 1 | 1e-4 | 16 | 0.3943 |
| C3 | Qwen2.5-1.5B | filtered | original | 1 | 2e-4 | 16 | 0.5139 |
| C4 | Qwen2.5-1.5B | filtered | original | 2 | 1.5e-4 | 16 | 0.5279 |
| C5 | Qwen2.5-1.5B | filtered | original | 5 | 1.5e-4 | 16 | 0.4400 |
| C6 | Qwen2.5-1.5B | full | original | 5 | 1.5e-4 | 16 | 0.3453 |
| C7 | Qwen2.5-1.5B | filtered | augmented-balanced | 2 | 1.5e-4 | 16 | 0.4345 |
| C8 | Qwen2.5-1.5B | filtered | original | 2 | 1.5e-4 | 32 | 0.4880 |

## Final Selected Submission Path

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter: `ec2_qwen25_1p5b_filtered_e2_lr15e5`
- Inference path: `python main.py --input <file> --output <file>`
- Verified released-validation score: `0.5511`
- Table score: `0.6314`
- Column score: `0.4708`

## Notes

- The best raw checkpoint before the final runtime verification was C4 at `0.5279`.
- The augmented-balanced data variant did not help and was not used in the final submission.
- The selected runtime path was verified on EC2 through the TA-facing `main.py` entrypoint.
