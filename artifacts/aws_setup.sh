set -euo pipefail

rm -rf "$HOME/miniforge"

wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O "$HOME/miniforge.sh"
bash "$HOME/miniforge.sh" -b -p "$HOME/miniforge"

source "$HOME/miniforge/bin/activate"
eval "$("$HOME/miniforge/bin/conda" shell.bash hook)"

conda config --set channel_priority strict
conda config --remove channels defaults 2>/dev/null || true
conda config --add channels conda-forge

mamba create -y -n cse234 python=3.12 pip

conda activate cse234

pip install -U pip
pip install uv ipykernel
uv pip install rapidfireai loguru trl peft

rapidfireai init
