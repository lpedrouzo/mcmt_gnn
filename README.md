# Multi-Object Multi-Camera tracking with Graph Neural Networks
This repo is currently under active development. I know documentation needs a little bit of housekeeping. Just hang tight, we will get there
once the solution is good enough. I need to graduate first, then we can take care of making the documentation pretty.

## Getting Started

Create a conda environment:

```
conda create --name mcmt_env python=3.9
conda activate mcmt_env
```

Download the SOLIDER [CITE] REID model:

```
gdown 'https://drive.google.com/uc?id=12UyPVFmjoMVpQLHN07tNh4liHUmyDqg8&export=download' -O models/reid/st_reid_weights.pth
```
