# GNN-Guided Structure-Aware QEM Mesh Simplification

Official implementation of the paper: **"GNN-Guided Structure-Aware QEM Mesh Simplification"**.

[![License: CC BY](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Introduction
This repository provides a mesh simplification framework that integrates Graph Neural Networks (GNN) with classical Quadric Error Metrics (QEM). By predicting geometric saliency using a hybrid structural representation network (spectral geometry + dual-branch architecture), our method prioritizes critical features through a dynamic soft modulation mechanism.

## Environment Setup

### 1. Create Conda Environment
```bash
# Create environment with Python 3.10
conda create -n GNNMesh python=3.10 -y
conda activate GNNMesh
```

### 2. Install Dependencies
```bash
# Install PyTorch with CUDA 12.1 support
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt
```

## Dataset
Ensure your dataset is organized in a relative path, for example:
```text
./data_tosc/
└── toscahires_obj/   # Original high-resolution OBJ files
```

## Usage

### 1. Training (Edge Importance Prediction)
To train the structural importance estimator:
```bash
python train_edge_importance.py \
    --device cuda \
    --epochs 50 \
    --pe_dim 16 \
    --use_far_graph 1 \
    --max_far_neighbors 12 \
    --vis_every 5 \
    --vis_index 0
```
The model will be saved to `./checkpoints/edge_gnn.pt`.

### 2. Mesh Simplification
To perform simplification on a batch of models (e.g., at a 0.05 ratio):
```bash
python batch_simplify.py \
    --input_dir ./data_tosc/toscahires_obj \
    --output_dir ./data_tosc/toscahires_gnn_results \
    --ckpt ./checkpoints/edge_gnn_tosc.pt \
    --device cuda \
    --ratio 0.05 \
    --stages 2 \
    --gate_c0 0.20 \
    --refresh_gate_each_stage 1 \
    --relaxation_power 2.0 \
    --max_valence 60 \
    --max_valence_increase 12 \
    --export_imp_map \
    --imp_map_dir ./data_tosc/toscahires_gnn_results/imp_maps
```

## Project Structure
- `train_edge_importance.py`: Training script for GNN model.
- `batch_simplify.py`: Main entry for simplifying OBJ meshes.
- `models.py`: Defines the dual-branch GNN architecture.
- `mesh_dataset.py`: Graph construction and dataset handling.
- `simplify_mesh.py`: Core logic for learning-guided edge collapse.


