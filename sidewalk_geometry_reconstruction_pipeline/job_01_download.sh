#!/bin/bash
#SBATCH --job-name="seg_depth_download"
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --account=your_account_here  # TODO: Update with your cluster account
#SBATCH --output=../data/download_%j.out
#SBATCH --error=../data/download_%j.err

echo "=========================================="
echo "Phase 1: Download Mapillary Data"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "=========================================="

module purge
# module load miniconda3  # Uncomment and modify to load your cluster's conda module if needed/4.12.0

ENV_PATH="your_conda_env_path"
echo "Activating environment: $ENV_PATH"
eval "$(conda shell.bash hook)"
conda activate "$ENV_PATH"

cd ../data

SECTION=${1:-boston_south_end}

echo "Downloading section: $SECTION"

python 01_download_data.py --sections "$SECTION" --resolution 2048

echo "=========================================="
echo "Done downloading $SECTION"
echo "End Time: $(date)"
echo "=========================================="
