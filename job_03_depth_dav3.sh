#!/bin/bash
#SBATCH --job-name="seg_depth_dav3"
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4000M
#SBATCH --account=your_account_here  # TODO: Update with your cluster account
#SBATCH --output=.//depth_dav3_%j.out
#SBATCH --error=.//depth_dav3_%j.err

set -e

echo "=========================================="
echo "Phase 3: DA3 Pose-Conditioned Depth"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "=========================================="

module purge
module load cuda/12.1
module load miniconda3/4.12.0

ENV_PATH="your_conda_env_path"
echo "Activating environment: $ENV_PATH"
eval "$(conda shell.bash hook)"
conda activate "$ENV_PATH"

# Add the DA3 source package to PYTHONPATH
export PYTHONPATH="./models/Depth-Anything-V3/src:$PYTHONPATH"

SECTION=${1:-boston_backbay}

cd ./

echo "=========================================="
echo "Starting Pose-Conditioned DA3 Depth for $SECTION"
echo "Start Time: $(date)"
echo "=========================================="

"$ENV_PATH/bin/python" 03_generate_depth_dav3.py "../../$SECTION" --window-size "${2:-4}"

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
