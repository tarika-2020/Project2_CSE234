import argparse
import json
import os


DEFAULT_CONFIGS = [
    {
        "name": "qwen25_1p5b_filtered_e2_lr15e5_r16",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "dataset_variant": "filtered",
        "learning_rate": 1.5e-4,
        "lora_r": 16,
        "num_train_epochs": 2.0,
    },
    {
        "name": "qwen25_1p5b_filtered_e1_lr15e5_r16",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "dataset_variant": "filtered",
        "learning_rate": 1.5e-4,
        "lora_r": 16,
        "num_train_epochs": 1.0,
    },
    {
        "name": "qwen25_1p5b_filtered_e3_lr15e5_r16",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "dataset_variant": "filtered",
        "learning_rate": 1.5e-4,
        "lora_r": 16,
        "num_train_epochs": 3.0,
    },
    {
        "name": "qwen25_1p5b_filtered_e2_lr10e5_r16",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "dataset_variant": "filtered",
        "learning_rate": 1e-4,
        "lora_r": 16,
        "num_train_epochs": 2.0,
    },
    {
        "name": "qwen25_1p5b_filtered_e2_lr20e5_r16",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "dataset_variant": "filtered",
        "learning_rate": 2e-4,
        "lora_r": 16,
        "num_train_epochs": 2.0,
    },
    {
        "name": "qwen25_1p5b_filtered_e2_lr15e5_r8",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "dataset_variant": "filtered",
        "learning_rate": 1.5e-4,
        "lora_r": 8,
        "num_train_epochs": 2.0,
    },
    {
        "name": "qwen25_1p5b_filtered_e2_lr15e5_r32",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "dataset_variant": "filtered",
        "learning_rate": 1.5e-4,
        "lora_r": 32,
        "num_train_epochs": 2.0,
    },
    {
        "name": "qwen25_1p5b_full_schema_e2_lr15e5_r16",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "dataset_variant": "full_schema",
        "learning_rate": 1.5e-4,
        "lora_r": 16,
        "num_train_epochs": 2.0,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./artifacts/experiment_configs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for config in DEFAULT_CONFIGS:
        path = os.path.join(args.out_dir, f"{config['name']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
