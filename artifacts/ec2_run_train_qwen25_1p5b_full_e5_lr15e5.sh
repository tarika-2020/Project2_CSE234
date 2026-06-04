set -euo pipefail

source "$HOME/miniforge/bin/activate"
eval "$("$HOME/miniforge/bin/conda" shell.bash hook)"
conda activate cse234

cd "$HOME/Project2_CSE234"
mkdir -p artifacts logs

python scripts/train_qlora.py \
  --train_file artifacts/prepared/train_full_schema.jsonl \
  --eval_file artifacts/prepared/validation_full_schema.jsonl \
  --output_dir artifacts/ec2_qwen25_1p5b_full_e5_lr15e5 \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --num_train_epochs 5 \
  --learning_rate 1.5e-4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 1024 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --save_steps 50 \
  --load_in_4bit \
  --bf16 \
  --config_out artifacts/ec2_qwen25_1p5b_full_e5_lr15e5_config.json
