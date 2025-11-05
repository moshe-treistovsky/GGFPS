#!/usr/bin/env python3
import os
import argparse
import numpy as np
from ase.io import read
from ase.io.extxyz import write_extxyz

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

def adaptive_fps_sweep(gradients, D, n_indices, gamma_start=-1.0, gamma_end=1.0, random_state=None):
    """
    Adaptive Furthest Point Sampling with a shifting gradient bias from low to high gradient norms.
    
    Parameters:
    - gradients: np.ndarray, shape (n_samples,)
        The gradient norms associated with each point.
    - D: np.ndarray, shape (n_samples, n_samples)
        The precomputed distance matrix between points.
    - n_indices: int
        The number of points to select.
    - gamma_start: float
        The initial gamma value (negative to bias towards low gradients).
    - gamma_end: float
        The final gamma value (positive to bias towards high gradients).
    - random_state: int or None
        Seed for random number generator.
        
    Returns:
    - selected_indices: np.ndarray, shape (n_indices,)
        Indices of the selected points.
    """
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = D.shape[0]
    selected_indices = []
    unselected_indices = list(range(n_samples))
    
    # Initialize min distances to infinity
    min_distances = np.full(n_samples, np.inf)
    
    # Compute initial gamma
    gamma_i = gamma_start  # At iteration i=0
    
    # Compute selection probabilities proportional to gradients ** gamma_i
    # Avoid zero gradients by adding a small epsilon
    epsilon = 1e-8
    probabilities = (gradients + epsilon) ** gamma_i
    probabilities /= probabilities.sum()
    
    # Select the initial index randomly based on probabilities
    initial_index = np.random.choice(n_samples, p=probabilities)
    
    selected_indices.append(initial_index)
    unselected_indices.remove(initial_index)
    
    # Initialize min distances with distances from the initial point
    min_distances = np.minimum(min_distances, D[initial_index])
    
    for i in range(1, n_indices):
        # Compute gamma_i for current iteration
        gamma_i = gamma_start + (gamma_end - gamma_start) * (i / (n_indices - 1))
        
        # Compute weighted scores
        # Avoid zero gradients by adding a small epsilon
        weighted_scores = ((gradients[unselected_indices] + epsilon) ** gamma_i) * min_distances[unselected_indices]
        
        # Select the point with the maximum weighted score
        next_index = unselected_indices[np.argmax(weighted_scores)]
        selected_indices.append(next_index)
        unselected_indices.remove(next_index)
        
        # Update min distances
        min_distances[unselected_indices] = np.minimum(min_distances[unselected_indices], D[next_index, unselected_indices])
        
    return np.array(selected_indices)

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

                ggfps_idx = adaptive_fps_sweep(rms_force_per_atom, 
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