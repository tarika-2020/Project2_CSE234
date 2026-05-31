import argparse
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold", default=os.path.join(ROOT, "validation_gold_schema_links.json"))
    parser.add_argument("--questions_input", default=os.path.join(ROOT, "validation_input.json"))
    parser.add_argument("--schemas_dir", default=os.path.join(ROOT, "schemas"))
    parser.add_argument("--per_question_out", default=os.path.join(ROOT, "artifacts", "per_question.csv"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.per_question_out), exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(ROOT, "eval.py"),
        "--predictions",
        args.predictions,
        "--gold",
        args.gold,
        "--schemas_dir",
        args.schemas_dir,
        "--questions_input",
        args.questions_input,
        "--per_question_out",
        args.per_question_out,
    ]
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
