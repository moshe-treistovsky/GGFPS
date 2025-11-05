# OQML-GNN: Pooled MACE descriptors + FPS / GGFPS sampling

This repository contains:
- `gen_MACE_desc_DMAT.py`: compute pooled MACE descriptors per frame and a full pairwise distance matrix.
- `gen_train_from_DMAT.py`: sample frames by FPS, uniform random sampling (URS), and GGFPS using an input distance matrix and a gradient proxy (RMS force per atom).
- `GGFPS.py`: the GGFPS sampling algorithm. 
- `KRR_GGFPS.py`: the entire GGFPS/KRR cross validation workflow
## Install (dev)
```bash
pip install -e .
