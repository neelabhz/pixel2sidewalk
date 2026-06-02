#!/bin/bash
#SBATCH --job-name=test_da3
#SBATCH --time=07:00:00
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=7800M
#SBATCH --gpus-per-task=1
#SBATCH --account=your_account_here  # TODO: Update with your cluster account
#SBATCH --output=.//test_da3_%j.out
#SBATCH --error=.//test_da3_%j.err

module load cuda/12.1
module load miniconda3/4.12.0
eval "$(conda shell.bash hook)"
conda activate your_conda_env_path

export PYTHONPATH="../models/Depth-Anything-V3/src:$PYTHONPATH"

cd ./
python test_sequences.py --phase all
