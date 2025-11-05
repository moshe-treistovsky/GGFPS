# oqml/core.py
from __future__ import annotations
from typing import List, Optional
import numpy as np



def ensure_rng(random_state: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(random_state)

def adaptive_fps_sweep(
    gradients,
    D,
    n_indices: int,
    gamma_start: float = -1.0,
    gamma_end: float = 1.0,
    random_state: Optional[int] = None,
):
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
    rng = ensure_rng(random_state)
    gradients = np.asarray(gradients).reshape(-1)
    n_samples = D.shape[0]
    if n_indices > n_samples:
        raise ValueError("n_indices cannot exceed number of samples in D.")
    selected_indices: List[int] = []
    unselected_indices = list(range(n_samples))

    # Distances to current set
    min_distances = np.full(n_samples, np.inf)

    # initial gamma and probabilities
    # Avoid zero gradients by adding a small epsilon
    epsilon = 1e-8
    gamma_i = gamma_start
    probs = (gradients + epsilon) ** gamma_i
    probs = probs / probs.sum()

    # Select the initial index randomly based on probabilities
    initial_index = int(rng.choice(n_samples, p=probs))
    selected_indices.append(initial_index)
    unselected_indices.remove(initial_index)

    # Initialize min distances with distances from the initial point
    min_distances = np.minimum(min_distances, D[initial_index])

    for i in range(1, n_indices):
        # Compute gamma_i for current iteration
        gamma_i = gamma_start + (gamma_end - gamma_start) * (i / (n_indices - 1))

        # weighted score = gradient^gamma * distances between selected index and unselected indices
        weighted_scores = ((gradients[unselected_indices] + epsilon) ** gamma_i) * min_distances[unselected_indices]

        next_index = unselected_indices[np.argmax(weighted_scores)]

        selected_indices.append(next_index)
        unselected_indices.remove(next_index)
        # update min distances
        min_distances[unselected_indices] = np.minimum(min_distances[unselected_indices], D[next_index, unselected_indices])
        
    return np.asarray(selected_indices, dtype=int)