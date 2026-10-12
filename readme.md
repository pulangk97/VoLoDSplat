# VoLoDSplat: Voxel-based Hierarchical Feed-forward Gaussian Splatting

Official implementation of **VoLoDSplat: Voxel-based Hierarchical Feed-forward Gaussian Splatting**, a SIGGRAPH Asia 2026 paper.

VoLoDSplat is a feed-forward framework for novel-view synthesis. It builds a hierarchical voxel-based scene representation and predicts 3D Gaussians from sparse input views. This repository currently provides the training and evaluation pipeline for the RealEstate10K (RE10K) dataset.

## Overview

- Feed-forward 3D Gaussian prediction from sparse views
- Hierarchical voxel-based scene representation
- Multi-scale Gaussian rendering and supervision
- RE10K training and benchmark evaluation
- Six-view evaluation using the fixed index in `assets/re10k_evaluation/evaluation_index_re10k.json`

## Installation

The code has been tested with Python 3.10, PyTorch 2.4.0, and CUDA 12.1.

```bash
conda create -n volodsplat python=3.10
conda activate volodsplat

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  xformers==0.0.27.post2 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### MinkowskiEngine

This repository includes a CUDA 12-compatible MinkowskiEngine source tree.

```bash
conda install -c conda-forge openblas
pip install ninja
cd MinkowskiEngine
python setup.py install
cd ..
```

### Depth Anything 3 and gsplat

Depth Anything 3 is included under `src/model/encoder/Depth-Anything-3`.

```bash
pip install -e src/model/encoder/Depth-Anything-3
pip install --no-build-isolation \
  git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

## Pretrained Models

Create the checkpoint directory and download the UniMatch and Depth Anything 3 weights:

```bash
mkdir -p pretrained
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth \
  -P pretrained
```

Download [DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL) and place it at:

```text
pretrained/
├── gmflow-scale1-things-e9887eda.pth
└── DA3-SMALL/
```

## Dataset

Prepare the preprocessed RealEstate10K dataset using the following layout:

```text
datasets/re10k/
├── train/
│   ├── *.torch
│   └── index.json
└── test/
    ├── *.torch
    └── index.json
```

The default scripts expect the dataset root to be `datasets/re10k`. You can change it by editing `dataset.roots` in the corresponding shell script or by passing a Hydra override.

## Training

Start RE10K training with:

```bash
bash train.sh
```

`train.sh` contains the default training configuration, including the dataset path, context-view count, output directory, pretrained UniMatch checkpoint, and Weights & Biases project name. Edit these values in the script as needed before launching a run.

Training outputs and checkpoints are written under `outputs/` by default.

## Evaluation

First set `checkpointing.pretrained_model` in `evaluation.sh` to the checkpoint you want to evaluate, then run:

```bash
bash evaluation.sh
```

The default evaluation uses:

- 6 context views
- 8 target views defined by the evaluation index
- `assets/re10k_evaluation/evaluation_index_re10k.json`
- Every 10th RE10K test chunk

Rendered images and other optional artifacts are disabled in the default script. Set the relevant `test.save_*` options in `evaluation.sh` to `true` if you want to save them.

## Configuration

VoLoDSplat uses [Hydra](https://hydra.cc/) for configuration. The provided shell scripts collect the full set of overrides used for the default experiments. To change an experiment, copy the relevant script and edit its overrides before launching it.

For reproducible experiments, we recommend keeping the launch script together with the resulting checkpoint.

## Acknowledgements

This project is primarily built upon [VolSplat](https://github.com/ziplab/VolSplat). We sincerely thank the VolSplat authors for their excellent work and open-source contribution. We also acknowledge [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [UniMatch](https://github.com/autonomousvision/unimatch), [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), and [gsplat](https://github.com/nerfstudio-project/gsplat).

## Citation

If you find this project useful, please cite the VoLoDSplat paper. The BibTeX entry will be added after the paper information is publicly available.
