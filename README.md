<h1 align="center"><a href="https://spatialailab.github.io/shape-of-you/"><img src="icon/soy.png" height="45" align="absmiddle" alt="icon"></a> Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild</h1>

<div align="center">
  Jiin Im, Sisung Liu, and Je Hyeong Hong
</div>

<p align="center">
  <b>Spatial AI Lab @ Hanyang University</b><br>
  CVPR 2026
</p>

<div align="center">
  <a href="https://spatialailab.github.io/shape-of-you/"><img src="https://img.shields.io/badge/Project-Page-blue?style=flat-square" alt="Project Page"></a>
  <a href="https://arxiv.org/pdf/2603.11618"><img src="https://img.shields.io/badge/arXiv-PDF-b31b1b?style=flat-square" alt="Paper"></a>
</div>

---

This repository is the official implementation of **"Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild"**.

This initial code release includes **zero-shot evaluation**, **Shape-of-You best checkpoint evaluation**, and **training code**.
The repository is currently undergoing cleanup and validation. We are actively verifying the installation process, evaluation pipeline, and training entrypoint across different environments. Additional fixes and documentation updates may be provided during this period.

## Introduction

We propose **Shape-of-You**, a novel approach to semantic correspondence using Fused Gromov-Wasserstein (FGW) optimal transport. Our method effectively captures both appearance and geometric structures for robust matching in-the-wild.

## Code Release

The current release is organized for:
- Preparing SPair-71k.
- Extracting DINOv2 + Stable Diffusion features.
- Extracting SAM masks.
- Lifting 3D points with VGGT.
- Running zero-shot Gromov-Wasserstein (GW) linearization evaluation.
- Running evaluation with the Shape-of-You best checkpoint.
- Launching the SPair-71k training pipeline.

The repository layout is:

- `src/eval/` - preprocessing and zero-shot evaluation:
  - `preprocess_map.py`: DINOv2 + SD feature extraction
  - `preprocess_mask_sam.py`: SAM mask extraction
  - `evaluation.py`: zero-shot GW correspondence evaluation
- `src/vggt/` - project-specific scripts for lifting 3D points with VGGT.
- `src/train/` - SPair-71k training code.
- `configs/eval/` - YAML config for zero-shot and Shape-of-You checkpoint evaluation.
- `configs/train/` - JSON config for training.
- `scripts/` - dataset download helper scripts.
- `../data/` - user-created directory for SPair-71k and all precomputed files.

All commands below assume this layout and are run from the repository root unless stated otherwise.

## Environment

We conduct all experiments with the following environment:

- Python 3.10
- CUDA 11.7
- PyTorch 2.0.1 and torchvision 0.15.2
- Linux (Ubuntu 20.04/22.04) with NVIDIA GPUs

Set up the environment with conda as follows:

```bash
conda create -n shapeofyou python=3.10
conda activate shapeofyou

# PyTorch + CUDA 11.7
pip install torch==2.0.1 torchvision==0.15.2

# Project dependencies
pip install -r requirements.txt
```

Third-party projects used by this repository are summarized in `THIRD_PARTY_NOTICES.md`.

## Dataset: SPair-71k

Place the SPair-71k dataset one level above the repository:

```bash
bash scripts/download_spair.sh
```

## Precomputation

These precomputations are required before zero-shot evaluation.

### DINOv2 + SD Feature Extraction

```bash
cd src/eval

python preprocess_map.py \
  --base_dir ../../../data/SPair-71k/JPEGImages/ \
  --dino --sd
```

This script computes dense DINOv2 + Stable Diffusion feature maps for all SPair-71k images and stores them under the internal feature directory used by evaluation.

### SAM Mask Extraction

```bash
cd src/eval

pip install git+https://github.com/facebookresearch/segment-anything.git
mkdir -p weight
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P weight

python preprocess_mask_sam.py
```

This script:
- Loads the SAM checkpoint from `weight/sam_vit_h_4b8939.pth`,
- Computes object masks for SPair-71k images, and
- Saves them to the mask directory used by evaluation.

### VGGT Point Extraction

Our point lifting scripts use VGGT from the official `facebookresearch/vggt` repository.

```bash
pip install git+https://github.com/facebookresearch/vggt.git

cd src/vggt
bash extract_point.sh
```

The `extract_point.py` and `extract_point.sh` files in this repository are project-specific wrappers for SPair-71k. The VGGT implementation itself is provided by the official repository above. This script lifts SPair-71k image pairs with VGGT and saves 3D point sets / geometry to disk.

## Zero-shot Evaluation

Evaluation is handled by `src/eval/evaluation.py`. The provided config runs zero-shot GW linearization without loading the Shape-of-You best checkpoint.

```bash
cd src/eval

python evaluation.py --config ../../configs/eval/spair.yaml
```

Conceptually, this mode:
1. Uses precomputed DINOv2 + SD feature maps and SAM masks.
2. Optionally uses VGGT-lifted geometry in the matching cost.
3. Computes a soft correspondence matrix through linearized Gromov-Wasserstein matching.
4. Converts the correspondence into keypoint matches and reports PCK and related metrics.

### Shape-of-You Best Checkpoint Evaluation

Download the Shape-of-You best checkpoint asset and place it under `checkpoints/`. The optional `shapeofyou_best.json` file records the training config for the released checkpoint.

```bash
mkdir -p checkpoints
# place shapeofyou_best.pth at checkpoints/shapeofyou_best.pth
```

Then run evaluation with the Shape-of-You aggregation network:

```bash
cd src/eval

python evaluation.py --config ../../configs/eval/shapeofyou_best.yaml
```

For zero-shot evaluation, use `../../configs/eval/spair.yaml`, which omits `LOAD` and keeps the dummy-network behavior.

## Training

The training entrypoint is provided for SPair-71k with precomputed DINOv2 + Stable Diffusion features, SAM masks, VGGT-lifted 3D points, and generated pseudo correspondences.

From the repository root:

```bash
python src/train/train_sc.py --config configs/train/shapeofyou.json
```

Training outputs are written under `saved/`. W&B logging is disabled by default in the provided training config.

## Roadmap

- [x] Release paper on arXiv
- [x] Release zero-shot evaluation code
- [x] Release Shape-of-You best checkpoint evaluation
- [x] Release training code

## Acknowledgements

We thank the authors of [GeoAware-SC](https://github.com/Junyi42/GeoAware-SC) for releasing their open-source implementation, which served as a helpful reference for semantic correspondence evaluation and geometry-aware analysis.

## Citation

```bibtex
@inproceedings{im2026shapeofyou,
  title={Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild},
  author={Im, Jiin and Liu, Sisung and Hong, Je Hyeong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
