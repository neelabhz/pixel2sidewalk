# Mapillary Sidewalk Detection (Inventory) Pipeline

This pipeline implements a topological approach to sidewalk detection. It downloads Mapillary metadata and vector segmentation masks for a specified area, processes the detections, and aggregates the sidewalk presence data onto an OpenStreetMap (OSM) road network graph.

## Setup & Requirements

### 1. Environment Configuration
You can recreate the exact conda environment needed to run this pipeline using the provided `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate sidewalk_inventory
```

### 2. Mapillary API Key
This pipeline requires a Mapillary Developer API Token to fetch metadata and images.
- **How to get it:** Create a free account at [Mapillary Developer](https://www.mapillary.com/developer), register an application, and copy your Client Token.
- **Configuration:** Set this token as an environment variable before running the script:
  ```bash
  export MAPILLARY_CLIENT_TOKEN="your_token_here"
  ```

---

## How to Run

### Step 0: Define Area & Fetch Metadata
Edit `0_setup_new_area.py` and set your `BBOX` and `AREA_NAME`.
```bash
python 0_setup_new_area.py
```
This fetches all Mapillary metadata and the OSM graph for the bounding box.

### Step 1: Download Vector Masks
Requires internet access. This script downloads the actual PNG segmentation masks from Mapillary APIs.
```bash
python 1_phase1_download.py <AREA_NAME>
```
*(Wait for this to finish, it can take time for large areas with many panoramas)*

### Step 2 & 3: Process & Aggregate
Once masks are downloaded, run the offline processing to extract sidewalk pixels and aggregate them to the road network (Topological Approach).
```bash
python 2_phase2_process.py
python 3_phase3_aggregate.py
```

*Note: If you are running on a cluster environment, you can submit the included `job_mapillary_offline.sh` script to run Step 2 and 3 automatically on a compute node.*
