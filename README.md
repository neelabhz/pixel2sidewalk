# Pixel2Sidewalk

Welcome to **Pixel2Sidewalk**, a repository containing computer vision and spatial pipelines for mapping and reconstructing sidewalk geometry from 2D street-level panoramas (Mapillary).

This repository is organized into two distinct, standalone methodologies:

## 1. Sidewalk Geometry Reconstruction Pipeline
This approach uses metric depth estimation (Depth-Anything-V3) and zero-shot segmentation (DINOv3) to physically project 2D image pixels into, continuous 3D GeoJSON polygons.

🔗 **[Go to Geometry Reconstruction Pipeline](./sidewalk_geometry_reconstruction_pipeline/README.md)**

## 2. Sidewalk Inventory (Topological) Pipeline
This approach uses segmentation masks to detect the presence of sidewalks and aggregately maps them onto an OpenStreetMap (OSM) road network graph. It is a scalable method for city-level inventory analysis.

🔗 **[Go to Sidewalk Inventory Pipeline](./sidewalk_inventory_pipeline/README.md)**

---

## Repository Directory Structure

When running these pipelines, maintain the following structure for your downloaded data and model weights. Both pipelines are configured to look for local data in a generic `./data/` folder and model weights in a `./models/` folder.

```text
pixel2sidewalk/
│
├── sidewalk_geometry_reconstruction_pipeline/
│   ├── 00_reproject_panoramas.py
│   ├── ... (code files)
│   └── README.md
│
├── sidewalk_inventory_pipeline/
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
 You must expose your Mapillary Developer token via your system environment variables before running the data download scripts:
```bash
export MAPILLARY_CLIENT_TOKEN="your_token_here"
```
