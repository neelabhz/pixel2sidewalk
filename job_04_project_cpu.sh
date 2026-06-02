#!/bin/bash
#SBATCH --job-name=proj_cpu
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3900M
#SBATCH --account=your_account_here  # TODO: Update with your cluster account
#SBATCH --output=.//proj_cpu_%j.out
#SBATCH --error=.//proj_cpu_%j.err

# module load miniconda3  # Uncomment and modify to load your cluster's conda module if needed

eval "$(conda shell.bash hook)"
conda activate depth_env

echo "=== Phase 4: Projection (CPU only) ==="
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo ""

SECTION=${1:-boston_backbay}

your_conda_env_path/bin/python -u .//04_project_geometry.py \
    --sections ./data/$SECTION \
    --depth-dir dav3_depth \
    --step 5.0 \
    --cam-dist 25.0 \
    --min-density 0.5

echo ""
echo "=== Phase 4 Complete ==="
echo "Output files:"
ls -lh .//${SECTION}_*.geojson 2>/dev/null
ls -lh .//${SECTION}_map.html 2>/dev/null
