# Pixel2Sidewalk

Welcome to **Pixel2Sidewalk**, a repository containing state-of-the-art computer vision and spatial pipelines for mapping and reconstructing sidewalk geometry from 2D street-level panoramas (Mapillary).

This repository is organized into two distinct, standalone methodologies:

## 1. Sidewalk Geometry Reconstruction Pipeline
This approach uses metric depth estimation (Depth-Anything-V3) and zero-shot segmentation (DINOv3) to physically project 2D image pixels into highly accurate, continuous 3D GeoJSON polygons. It is excellent for modeling exact variable-width sidewalk geometries and generating 3D surfaces.

🔗 **[Go to Geometry Reconstruction Pipeline](./sidewalk_geometry_reconstruction_pipeline/README.md)**

## 2. Sidewalk Inventory (Topological) Pipeline
This approach uses segmentation masks to detect the presence of sidewalks and aggregately maps them onto an OpenStreetMap (OSM) road network graph. It is highly scalable and perfect for topological connectivity mapping and macroscopic city inventory analysis.

🔗 **[Go to Sidewalk Inventory Pipeline](./sidewalk_inventory_pipeline/README.md)**

---

## Repository Directory Structure

When running these pipelines, it is highly recommended to maintain the following structure for your downloaded data and model weights. Both pipelines are configured to look for local data in a generic `./data/` folder and model weights in a `./models/` folder.

```text
pixel2sidewalk/
│
├── sidewalk_geometry_reconstruction_pipeline/
│   ├── README.md                  <-- Setup & run instructions for Geometry Pipeline
│   ├── environment.yml
│   ├── 00_reproject_panoramas.py
│   ├── ...
│   └── data/                      <-- (Auto-created) Raw panoramas and output .geojson
│
├── sidewalk_inventory_pipeline/
│   ├── README.md                  <-- Setup & run instructions for Inventory Pipeline
│   ├── environment.yml
│   ├── 0_setup_new_area.py
│   ├── ... (code files)
│   └── README.md
│
├── models/                         <-- Place downloaded .pth weights here
│   ├── DA3NESTED-GIANT-LARGE-1.1/
│   └── ...
│
└── data/                           <-- Mapillary image data will download here
    ├── boston_backbay/
    └── de_pijp/
```

### API Keys
Both pipelines require a Mapillary Developer API Token. Register for free at [mapillary.com/developer](https://www.mapillary.com/developer), create an application, and copy your Client Token. Then set it as an environment variable before running:
```bash
export MAPILLARY_CLIENT_TOKEN="your_token_here"
```

### Model Weights
The **Geometry Reconstruction Pipeline** requires two pre-trained models. Create a `models/` folder at the root of this repository and place the weights inside:

| Model | Access | Link |
|-------|--------|------|
| **DINOv3** | Gated (request via Meta) | [GitHub](https://github.com/facebookresearch/dinov3) · [Meta Downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) · [HuggingFace (gated)](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m) |
| **Depth-Anything-V3** | Open | [GitHub](https://github.com/bytedance-seed/depth-anything-3) |

> **Note:** DINOv3 weights are gated and must be requested through Meta's platform or HuggingFace. Depth-Anything-V3 weights are freely available.
