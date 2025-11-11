# GGFPS for KRR and MACE
Gradient Guided Furthest Point Sampling is an extension of the Furthest Point Sampling algorithm for use in labeled data with labeled gradients (or forces). 
GGFPS biases FPS to select points that have a tuneably diverse range of gradient values, which is shown to be useful both in selecting training sets for both analytical functions and molecular dynamics datasets.
More information can be found at [text](https://arxiv.org/abs/2510.08906) 

This repository contains:
- `gen_MACE_desc_DMAT.py`: compute pooled MACE descriptors per frame and a full pairwise distance matrix.
- `gen_train_from_DMAT.py`: sample frames by FPS, uniform random sampling (URS), and GGFPS using an input distance matrix and a gradient proxy (RMS force per atom).
- `GGFPS.py`: the GGFPS sampling algorithm. 
- `KRR_GGFPS.py`: the entire GGFPS/KRR cross validation workflow
## Install (dev)
```bash
pip install -e .
