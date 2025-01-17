#!/bin/sh

#SBATCH --nodes=1
#SBATCH --job-name=lossTest
#SBATCH --time=78:00:00
#SBATCH --mem=500000


nvidia-smi

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

python -m venv ltpEnvironment
source ltpEnvironment/bin/activate
pip install -r requirements.txt

PYTHON_SCRIPT="./main.py"

python3 "$PYTHON_SCRIPT"