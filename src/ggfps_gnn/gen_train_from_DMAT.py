#!/usr/bin/env python3
import os
import argparse
import numpy as np
from ase.io import read
from ase.io.extxyz import write_extxyz
from GGFPS import adaptive_ggfps_sweep

# your existing FPS implementation
def fps_distance_matrix(D, K, init_idx=None, random_state=None):
    N = D.shape[0]
    rng = np.random.RandomState(random_state)
    if init_idx is None:
        init_idx = rng.randint(N)
    selected = [init_idx]
    min_dists = D[init_idx].copy()
    for _ in range(1, K):
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)
        min_dists = np.minimum(min_dists, D[next_idx])
    return selected

# uniform random sampling
def urs(N, K, random_state=None):
    rng = np.random.RandomState(random_state)
    return rng.choice(N, size=K, replace=False).tolist()

def load_extxyz(file_path):
    frames = read(file_path, index=':', format='extxyz')
    return frames

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Sample frames by FPS, GGFPS & URS and write new .extxyz files"
    )
    p.add_argument('--dmat',     required=True,
                   help="Path to saved NxN .npy distance matrix")
    p.add_argument('--traj',     required=True,
                   help="Path to input .extxyz trajectory")
    p.add_argument('--outdir',   required=True,
                   help="Directory to write sampled .extxyz files")
    p.add_argument('--seed',     type=int, default=0,
                   help="Random seed for FPS (if init random) and URS")
    p.add_argument('--n_bootstrap',type=int, default=10,
                help="Number of bootstrap replicates per method and size")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load data
    D      = np.load(args.dmat)       # (N,N)
    frames = load_extxyz(args.traj)   # list of Atoms
    N       = len(frames)
    name = os.path.splitext(os.path.basename(args.traj))[0]


    # forces: list of (natoms, 3) arrays
    forces = [f.get_array('FORCES') for f in frames]
    force_norms = np.array([np.linalg.norm(f) for f in forces])
    # number of atoms per frame
    natoms = np.array([f.shape[0] for f in forces], dtype=float)
    # per-frame sum of squared forces over atoms and xyz
    sum_sq = np.array([np.sum(F**2) for F in forces], dtype=float)
    # per-atom RMS force: sqrt( (1/natoms) * sum_i ||f_i||^2 )
    rms_force_per_atom = np.sqrt(sum_sq / natoms)

    # import your GGFPS routine

    sizes = [100, 250, 500, 750, 1000]
    grad_biases = np.linspace(0, 2, 5)
    
    for b in range(args.n_bootstrap):
        bs_seed = args.seed + b*100
        for K in sizes:
            # 1) FPS
            fps_idx = fps_distance_matrix(D, K, random_state=bs_seed)
            # 2) URS
            urs_idx = urs(N, K, random_state=bs_seed + 1)
            # 3) GGFPS

            for gb_i, gb in enumerate(grad_biases):

                ggfps_idx = adaptive_ggfps_sweep(rms_force_per_atom, 
                                               D, 
                                               K, 
                                               -1*gb, 
                                               1*gb,
                                               random_state=bs_seed + 2)

                out_file = os.path.join(
                    args.outdir,
                    f"{name}_GGFPS_K{K}_bias{gb_i}_b{b}.xyz"
                )
                sampled = [frames[i] for i in ggfps_idx]
                for a in sampled:
                    if 'FORCES' not in a.arrays and 'forces' in a.arrays:
                        a.set_array('FORCES', a.get_array('forces'))
                    if 'ENERGY' not in a.info and 'energy' in a.info:
                        a.info['ENERGY'] = a.info['energy']
                with open(out_file, 'w') as f:
                    write_extxyz(f, sampled,
                                 columns=None,
                                 write_info=True,
                                 write_results=False,
                                 plain=False)
                print(f"Wrote GGFPS K={K} bias={gb_i} bs={b} → {out_file}")

            # Write FPS
            out_fps = os.path.join(args.outdir, f"{name}_FPS_K{K}_b{b}.xyz")
            sampled_fps = [frames[i] for i in fps_idx]
            for a in sampled_fps:
                if 'FORCES' not in a.arrays and 'forces' in a.arrays:
                    a.set_array('FORCES', a.get_array('forces'))
                if 'ENERGY' not in a.info and 'energy' in a.info:
                    a.info['ENERGY'] = a.info['energy']
            with open(out_fps, 'w') as f:
                write_extxyz(f, sampled_fps,
                             columns=None,
                             write_info=True,
                             write_results=False,
                             plain=False)
            print(f"Wrote FPS K={K} bs={b} → {out_fps}")

            # Write URS
            out_urs = os.path.join(args.outdir, f"{name}_URS_K{K}_b{b}.xyz")
            sampled_urs = [frames[i] for i in urs_idx]
            for a in sampled_urs:
                if 'FORCES' not in a.arrays and 'forces' in a.arrays:
                    a.set_array('FORCES', a.get_array('forces'))
                if 'ENERGY' not in a.info and 'energy' in a.info:
                    a.info['ENERGY'] = a.info['energy']
            with open(out_urs, 'w') as f:
                write_extxyz(f, sampled_urs,
                             columns=None,
                             write_info=True,
                             write_results=False,
                             plain=False)
            print(f"Wrote URS K={K} bs={b} → {out_urs}")