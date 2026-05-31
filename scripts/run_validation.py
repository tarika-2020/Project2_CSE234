import argparse
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(ROOT, "validation_input.json"))
    parser.add_argument("--output", default=os.path.join(ROOT, "artifacts", "validation_predictions.json"))
    parser.add_argument("--schemas_dir", default=os.path.join(ROOT, "schemas"))
    parser.add_argument("--adapter_dir", default=os.path.join(ROOT, "adapter"))
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    predict_cmd = [
        sys.executable,
        os.path.join(ROOT, "main.py"),
        "--input",
        args.input,
        "--output",
        args.output,
        "--schemas_dir",
        args.schemas_dir,
        "--adapter_dir",
        args.adapter_dir,
        "--base_model",
        args.base_model,
    ]
    eval_cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "eval_predictions.py"),
        "--predictions",
        args.output,
    ]
    predict_code = subprocess.call(predict_cmd, cwd=ROOT)
    if predict_code != 0:
        raise SystemExit(predict_code)
    raise SystemExit(subprocess.call(eval_cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
