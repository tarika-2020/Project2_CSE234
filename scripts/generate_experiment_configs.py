import argparse
import json
import os


DEFAULT_CONFIGS = [
    {"name": "baseline_filtered", "dataset_variant": "filtered", "learning_rate": 2e-4, "lora_r": 16, "num_train_epochs": 3.0},
    {"name": "baseline_full_schema", "dataset_variant": "full_schema", "learning_rate": 2e-4, "lora_r": 16, "num_train_epochs": 3.0},
    {"name": "lr_low", "dataset_variant": "filtered", "learning_rate": 1e-4, "lora_r": 16, "num_train_epochs": 3.0},
    {"name": "lr_high", "dataset_variant": "filtered", "learning_rate": 3e-4, "lora_r": 16, "num_train_epochs": 3.0},
    {"name": "rank_small", "dataset_variant": "filtered", "learning_rate": 2e-4, "lora_r": 8, "num_train_epochs": 3.0},
    {"name": "rank_large", "dataset_variant": "filtered", "learning_rate": 2e-4, "lora_r": 32, "num_train_epochs": 3.0},
    {"name": "epochs_short", "dataset_variant": "filtered", "learning_rate": 2e-4, "lora_r": 16, "num_train_epochs": 2.0},
    {"name": "epochs_long", "dataset_variant": "filtered", "learning_rate": 2e-4, "lora_r": 16, "num_train_epochs": 5.0},
    {"name": "prompt_variant", "dataset_variant": "filtered", "learning_rate": 2e-4, "lora_r": 16, "num_train_epochs": 3.0, "prompt_variant": "compact_instruction"},
    {"name": "wider_filter", "dataset_variant": "filtered", "learning_rate": 2e-4, "lora_r": 16, "num_train_epochs": 3.0, "filter_variant": "wide"},
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
