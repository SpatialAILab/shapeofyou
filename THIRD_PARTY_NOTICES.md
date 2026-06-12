# Third-Party Notices

This repository does not vendor the full source code or model weights of the following third-party projects. They are installed or downloaded by the user through the commands documented in `README.md`.

## Segment Anything

- Project: Segment Anything
- Repository: https://github.com/facebookresearch/segment-anything
- Usage: installed as a dependency for SAM-based mask extraction.
- Checkpoint: downloaded from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth by the user.

## VGGT

- Project: VGGT
- Repository: https://github.com/facebookresearch/vggt
- Usage: installed as a dependency for 3D point extraction.

The files under `src/vggt/` in this repository are project-specific wrappers for SPair-71k extraction and are not intended to redistribute the upstream VGGT implementation.
