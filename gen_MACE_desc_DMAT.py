#!/usr/bin/env python3
import argparse
import json
import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator


def pool_mean_over_atoms(desc: np.ndarray) -> np.ndarray:
    # desc shape: (n_atoms, feat)
    return desc.mean(axis=0)

def pool_element_scatter_mean(zs_frame: np.ndarray,
                              desc: np.ndarray,
                              elem_order: list[int]) -> np.ndarray:
    # For each element z in elem_order: mean of rows with that z, else zeros
    feat = desc.shape[1]
    chunks = []
    for z in elem_order:
        mask = (zs_frame == z)
        if np.any(mask):
            chunks.append(desc[mask].mean(axis=0))
        else:
            chunks.append(np.zeros(feat, dtype=desc.dtype))
    return np.concatenate(chunks, axis=0)

def main():
    p = argparse.ArgumentParser(
        description="Compute pooled MACE descriptors and a full distance matrix."
    )
    p.add_argument("--model",    required=True, help="Path to .model file")
    p.add_argument("--traj",     required=True, help="Path to XYZ trajectory")
    p.add_argument("--out_desc", required=True, help="Output .npy of pooled descriptors")
    p.add_argument("--out_dmat", required=True, help="Output .npy of pairwise distances")
    p.add_argument("--device",   default="cuda", choices=["cpu","cuda"], help="cdist device")
    p.add_argument("--pool",     default="mean",
                   choices=["mean", "elem-mean", "element-mean", "scatter"],
                   help="Pooling: mean over atoms or per-element scatter-mean")
    p.add_argument("--save-meta", default=None,
                   help="Optional path to save JSON metadata about pooling/elements.")
    args = p.parse_args()

    pool_mode = args.pool
    if pool_mode == "scatter":
        pool_mode = "elem-mean"
    if pool_mode == "element-mean":
        pool_mode = "elem-mean"

    # 1) Load model & trajectory
    calc = MACECalculator(model_paths=args.model, device="cuda" if torch.cuda.is_available() else "cpu")
    frames = read(args.traj, ":")

    # 2) Peek at feature dimension from first frame
    first_desc = calc.get_descriptors(frames[0])  # (n_atoms, feat)
    feat_dim = int(first_desc.shape[1])

    # 3) Establish element order if needed
    elem_order = None
    if pool_mode == "elem-mean":
        # Infer from the whole trajectory to keep length consistent within this file
        all_z = set()
        for at in frames:
            all_z.update(np.unique(at.get_atomic_numbers()).tolist())
        elem_order = sorted(int(z) for z in all_z)

    # 4) Compute pooled descriptors per frame
    pooled = []
    for atoms in frames:
        desc = calc.get_descriptors(atoms)  # (n_atoms, feat)
        if pool_mode == "mean":
            v = pool_mean_over_atoms(desc)
        else:
            zs = atoms.get_atomic_numbers()
            v = pool_element_scatter_mean(zs, desc, elem_order)
        pooled.append(v.astype(np.float32))

    pooled = np.vstack(pooled)  # shape: (n_frames, feat) or (n_frames, len(elem_order)*feat)
    np.save(args.out_desc, pooled)
    print(f"Saved pooled descriptors {pooled.shape} → {args.out_desc}")

    # 5) (Optional) save metadata for reproducibility
    if args.save_meta:
        meta = {
            "pool": pool_mode,
            "feat_dim": feat_dim,
            "elements_order_Z": elem_order if elem_order is not None else None,
            "traj": args.traj,
            "model": args.model,
        }
        with open(args.save_meta, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved meta → {args.save_meta}")

    # 6) Compute full pairwise L2 distances (careful: O(N^2) memory)
    N = pooled.shape[0]
    est_bytes = 4 * N * N  # float32
    est_gb = est_bytes / (1024**3)
    print(f"Computing torch.cdist on ({N},{pooled.shape[1]}) … estimated D size ≈ {est_gb:.2f} GiB")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = "cpu"

    X = torch.as_tensor(pooled, dtype=torch.float32, device=device)
    with torch.no_grad():
        D = torch.cdist(X, X)  # (N, N)
    D_cpu = D.cpu().numpy()
    np.save(args.out_dmat, D_cpu)
    print(f"Saved distance matrix {D_cpu.shape} → {args.out_dmat}")

if __name__ == "__main__":
    main()