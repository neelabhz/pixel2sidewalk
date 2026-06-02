#!/bin/bash
#SBATCH --job-name="seg_depth_segment"
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4000M
#SBATCH --account=your_account_here  # TODO: Update with your cluster account
#SBATCH --output=.//segment_%j.out
#SBATCH --error=.//segment_%j.err

echo "=========================================="
echo "Phase 2: DINOv3 Segmentation"
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

cd ./

SECTION=${1:-boston_backbay}
SECTION_DIR="../../${SECTION}"

echo "Segmenting section: $SECTION"

"$ENV_PATH/bin/python" 02_segment_masks.py "$SECTION_DIR" --batch-size 4

echo "=========================================="
echo "Done segmenting $SECTION"
echo "End Time: $(date)"
echo "=========================================="
