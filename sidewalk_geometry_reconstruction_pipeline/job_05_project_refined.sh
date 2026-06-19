#!/bin/bash
#SBATCH --job-name="seg_depth_project"
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --account=cluster_account
#SBATCH --output=.//project_%j.out
#SBATCH --error=.//project_%j.err

set -e

echo "=========================================="
echo "Phase 5: Refined Sidewalk Geometry Projection"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "=========================================="

module purge
# module load miniconda3  # Uncomment and modify to load your cluster's conda module if needed/4.12.0

ENV_PATH="/path/to/conda/env"
echo "Activating environment: $ENV_PATH"
eval "$(conda shell.bash hook)"
conda activate "$ENV_PATH"

cd ./

SECTION=${1:-boston_backbay}
DEPTH_DIR=${2:-dav3_depth}

echo "Projecting section: $SECTION with depth directory: $DEPTH_DIR"

"$ENV_PATH/bin/python" 05_project_refined_geometry.py \
    --sections "../$SECTION" \
    --depth-dir "$DEPTH_DIR" \
    --step 5.0 \
    --cam-dist 25.0 \
    --min-density 1.5

echo "=========================================="
echo "Done! Output: ${SECTION}_polygons.geojson"
echo "End Time: $(date)"
echo "=========================================="
