#!/bin/bash
#SBATCH --job-name=mapillary_offline
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --account=your_account_here  # TODO: Update with your cluster account
#SBATCH --time=03:00:00
#SBATCH --output=offline_%j.out
#SBATCH --error=offline_%j.err

# NOTE: Before running, ensure you have initialized the area (Phase 0)
# and downloaded the masks (Phase 1) on the login node.

module purge
eval "$(conda shell.bash hook)"
conda activate your_conda_env_path

AREA="amsterdam_zuidas_wide"
AREA_DIR="./${AREA}"

echo "== STARTING OFFLINE PIPELINE ON CLUSTER =="

echo "-----------------------------------"
echo "[STEP 2] Process Points Maps"
python 2_phase2_process.py ${AREA_DIR}

echo "-----------------------------------"
echo "[STEP 3A] Aggregate Topological Centerlines (Full Blocks)"
python 3_phase3_aggregate.py ${AREA_DIR} --osm-file ${AREA_DIR}/${AREA}.graphml

echo "-----------------------------------"
echo "[STEP 3B] Aggregate Topological Centerlines (5m Micro-Segments)"
python 3_phase3_aggregate.py ${AREA_DIR} --osm-file ${AREA_DIR}/${AREA}.graphml --chunk-size-m 5.0

echo "== PIPELINE COMPLETE =="
