# Pixel2Sidewalk: Geometry Reconstruction Pipeline

This repository contains a complete end-to-end pipeline to reconstruct 3D sidewalk geometries (polygons and centrelines) from 2D street-level panoramas. 

## Pipeline Overview

The pipeline operates in four distinct phases:
1. **Download Data:** Fetch 360-degree panoramas for a given bounding box via the Mapillary API.
2. **Segmentation:** Process panoramas using DINOv3 (Detectron2) to generate binary masks isolating sidewalks.
3. **Depth Estimation:** Predict metric depth maps using Depth-Anything-V3, skipping images that do not contain sidewalks.
4. **Geometry Projection:** Reproject masked depth pixels into 3D world coordinates (EPSG:4326/EPSG:28992) and generate highly accurate spatial polygons.

## Setup & Requirements

### 1. Environment Configuration
You can recreate the exact conda environment needed to run this pipeline by using the provided `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate pixel2sidewalk
```

### 2. Mapillary API Key
Phase 1 requires a Mapillary Developer API Token to download imagery.
- **How to get it:** Create a free account at [Mapillary Developer](https://www.mapillary.com/developer), register an application, and copy your Client Token.
- **Configuration:** Set this token as an environment variable before running the script:
  ```bash
  export MAPILLARY_CLIENT_TOKEN="your_token_here"
  ```

### 3. Model Weights
You will need to download the pre-trained weights for the two computer vision models utilized in the pipeline.

| Model | Purpose | Download Link | Placement Directory |
|-------|---------|---------------|---------------------|
| **DINOv3** | Sidewalk Masking | [DINOv3 HuggingFace](https://github.com/IDEA-Research/DINOv3) | `./models/` |
| **Depth-Anything-V3** | Depth Estimation | [DA3 GitHub Repository](https://github.com/LiheYoung/Depth-Anything-V3) | `./models/Depth-Anything-V3/` |

*Ensure you download the `DA3NESTED-GIANT-LARGE-1.1` weights for maximum accuracy.*

## Running the Pipeline

Execute the scripts sequentially for a specific bounding box (referred to as a `SECTION` in the code):

1. `python 01_download_data.py` (Downloads imagery for configured bounding boxes)
2. `python 02_segment_masks.py` (Runs DINOv3 segmentation)
3. `python 03_generate_depth_dav3.py` (Runs DA3 depth estimation)
4. `python 04_project_geometry.py --sections ./data/your_section` (Projects to `.geojson`)

Alternatively, if you are running on a SLURM cluster, you can submit the included `job_*.sh` batch scripts.
