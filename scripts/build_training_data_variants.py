import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from schema_linking.augmentation import (
    build_augmented_records,
    build_balanced_records,
    summarize_training_mix,
)


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=os.path.join(ROOT, "train.json"))
    parser.add_argument(
        "--out_dir",
        default=os.path.join(ROOT, "artifacts", "custom_training_data"),
    )
    parser.add_argument("--min_examples_per_db", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.train, encoding="utf-8") as f:
        train_records = json.load(f)

    augmented = build_augmented_records(train_records)
    balanced = build_balanced_records(train_records, min_examples_per_db=args.min_examples_per_db)
    augmented_balanced = build_balanced_records(augmented, min_examples_per_db=args.min_examples_per_db)

    outputs = {
        "augmented_train.json": augmented,
        "balanced_train.json": balanced,
        "augmented_balanced_train.json": augmented_balanced,
        "training_mix_summary.json": {
            "original_count": len(train_records),
            "augmented_count": len(augmented),
            "balanced_count": len(balanced),
            "augmented_balanced_count": len(augmented_balanced),
            "augmented_summary": summarize_training_mix(augmented),
            "balanced_summary": summarize_training_mix(balanced),
            "augmented_balanced_summary": summarize_training_mix(augmented_balanced),
        },
    }

    for filename, payload in outputs.items():
        path = os.path.join(args.out_dir, filename)
        write_json(path, payload)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
