# Sidewalk Inventory Pipeline (Mapillary Detect)

This pipeline handles the extraction of topological sidewalk inventory maps by downloading and processing Mapillary segmentation masks to build an aggregated graph.

## Workflow

1. **Setup New Area** (`0_setup_new_area.py`): Define your target bounding box (`BBOX`) and `AREA_NAME`. This script queries OSM (OpenStreetMap) and prepares the directory structure.
2. **Download Masks** (`1_phase1_download.py`): Fetches the raw 2D pixel-level segmentation masks from the Mapillary API.
3. **Process Images** (`2_phase2_process.py`): Processes the masks to extract the sidewalk pixels.
4. **Aggregate** (`3_phase3_aggregate.py`): Aggregates the 2D detections into a topological spatial graph network.

## Setup

1. **Environment:** Create the Conda environment using `conda env create -f environment.yml` and activate it.
2. **API Key:** Set your Mapillary Developer API Token as an environment variable to allow image downloads:
   ```bash
   export MAPILLARY_CLIENT_TOKEN="your_token_here"
   ```
3. **Run:** Execute the scripts sequentially starting with `0_setup_new_area.py`.
