import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from schema_linking.data_prep import prepare_examples, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=os.path.join(ROOT, "train.json"))
    parser.add_argument("--validation", default=os.path.join(ROOT, "validation.json"))
    parser.add_argument("--schemas_dir", default=os.path.join(ROOT, "schemas"))
    parser.add_argument("--out_dir", default=os.path.join(ROOT, "artifacts", "prepared"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.train, encoding="utf-8") as f:
        train_records = json.load(f)
    with open(args.validation, encoding="utf-8") as f:
        validation_records = json.load(f)

    variants = [
        ("filtered", True),
        ("full_schema", False),
    ]
    for name, use_filtered in variants:
        train_out = os.path.join(args.out_dir, f"train_{name}.jsonl")
        val_out = os.path.join(args.out_dir, f"validation_{name}.jsonl")
        write_jsonl(train_out, prepare_examples(train_records, args.schemas_dir, use_filtered))
        write_jsonl(val_out, prepare_examples(validation_records, args.schemas_dir, use_filtered))
        print(f"Wrote {train_out}")
        print(f"Wrote {val_out}")


if __name__ == "__main__":
    main()
