# oqml/core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

Array = np.ndarray

def rmse(a: Array, b: Array) -> float:
    """Root-mean-square error."""
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def stable_cholesky_solve(K: Array, y: Array) -> Array:
    """
    Solve K alpha = y assuming K is SPD(ish).
    Tries Cholesky then falls back to np.linalg.solve with a small jitter.
    """
    try:
        L = np.linalg.cholesky(K)
        # Solve L L^T alpha = y
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return alpha
    except np.linalg.LinAlgError:
        jitter = 1e-12 * max(1.0, float(np.trace(K)) / K.shape[0])
        K2 = K + np.eye(K.shape[0]) * jitter
        return np.linalg.solve(K2, y)

def ensure_rng(random_state: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(random_state)

def adaptive_fps_sweep(
    gradients: Array,
    D: Array,
    n_indices: int,
    gamma_start: float = -1.0,
    gamma_end: float = 1.0,
    random_state: Optional[int] = None,
) -> Array:
    """
    Adaptive FPS with smoothly varying gradient bias from gamma_start to gamma_end.

    gradients: shape (n_samples,) or (n_samples, 1)
    D: shape (n_samples, n_samples) distance matrix on the *candidate* set
    n_indices: number of selections to make (<= n_samples)
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

@dataclass
class Bounds:
    width_bounds: Union[Tuple[float, float], List[float]]
    reg_bounds: Union[Tuple[float, float], List[float]]
    search_density: Optional[int] = None  # required only if width_bounds is tuple

class Cross_Validation:
    """
    Kernel ridge regression cross-validation for global kernels.
    k_fun(Xa, Xb, width) -> K (len(A) x len(B))
    """
    def __init__(
        self,
        X: Array,
        Y: Array,
        train_val_inds: Array,
        test_inds: Array,
        bounds: Dict,
        random_state: Optional[int] = None,
        solver: Callable[[Array, Array], Array] = stable_cholesky_solve,
        error_fn: Callable[[Array, Array], float] = rmse,
    ):
        self.X = X
        self.Y = Y
        self.train_val_inds = np.asarray(train_val_inds, dtype=int)
        self.test_inds = np.asarray(test_inds, dtype=int)
        self.bounds = Bounds(**bounds)
        self.rng = ensure_rng(random_state)
        self.solver = solver
        self.error_fn = error_fn

    def k_fold_split(self, k: int) -> Tuple[List[Array], List[Array]]:
        perm = self.rng.permutation(self.train_val_inds)
        folds = np.array_split(perm, k)
        train_folds, val_folds = [], []
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i])
            train_folds.append(train_idx)
            val_folds.append(val_idx)
        return train_folds, val_folds

    def _width_grid(self) -> List[float]:
        wb = self.bounds.width_bounds
        if isinstance(wb, tuple):
            if not self.bounds.search_density:
                raise ValueError("search_density is required when width_bounds is a tuple.")
            return list(np.logspace(np.log10(wb[0]), np.log10(wb[1]), self.bounds.search_density))
        return list(wb)

    def _reg_grid(self) -> List[float]:
        rb = self.bounds.reg_bounds
        if isinstance(rb, tuple):
            return list(np.logspace(np.log10(rb[0]), np.log10(rb[1]), 4))
        return list(rb)

    def _validate(
        self,
        k_fun: Callable[[Array, Array, float], Array],
        width: float,
        reg: float,
        train_Y: Array,
        val_Y: Array,
        train_X: Array,
        val_X: Array,
    ) -> float:
        Ktt = k_fun(train_X, train_X, width)
        Kvt = k_fun(val_X,   train_X, width)
        Ktt[np.diag_indices_from(Ktt)] += reg
        alpha = self.solver(Ktt, train_Y)
        pred_val = Kvt @ alpha
        pred_trn = Ktt @ alpha
        return 0.5 * (self.error_fn(train_Y, pred_trn) + self.error_fn(val_Y, pred_val))

    def grid_search(
        self,
        k_fun: Callable[[Array, Array, float], Array],
        train_folds: List[Array],
        val_folds: List[Array],
    ) -> Dict[str, float]:
        widths = self._width_grid()
        regs   = self._reg_grid()
        best = {"width": None, "reg": None, "mae": np.inf}
        for w in widths:
            for r in regs:
                maes = []
                for tr_idx, vl_idx in zip(train_folds, val_folds):
                    trX, vlX = self.X[tr_idx], self.X[vl_idx]
                    trY, vlY = self.Y[tr_idx], self.Y[vl_idx]
                    maes.append(self._validate(k_fun, w, r, trY, vlY, trX, vlX))
                avg = float(np.mean(maes))
                if avg < best["mae"]:
                    best = {"width": float(w), "reg": float(r), "mae": avg}
        return {"width": best["width"], "reg": best["reg"]}

    def evaluate(
        self,
        k_fun: Callable[[Array, Array, float], Array],
        best_params: Dict[str, float],
        train_Y: Array,
        test_Y: Array,
        train_X: Array,
        test_X: Array,
    ) -> Tuple[float, Array]:
        w, r = best_params["width"], best_params["reg"]
        Ktt = k_fun(train_X, train_X, w)
        Kqt = k_fun(test_X,  train_X, w)
        Ktt[np.diag_indices_from(Ktt)] += r
        alpha = self.solver(Ktt, train_Y)
        pred = Kqt @ alpha
        mae = float(np.mean(np.abs(test_Y - pred)))
        return mae, pred

    def build(self, k_fun: Callable[[Array, Array, float], Array], num_folds: int) -> Dict:
        tr_folds, vl_folds = self.k_fold_split(num_folds)
        best = self.grid_search(k_fun, tr_folds, vl_folds)
        trX = self.X[self.train_val_inds]; trY = self.Y[self.train_val_inds]
        teX = self.X[self.test_inds];      teY = self.Y[self.test_inds]
        mae, pred = self.evaluate(k_fun, best, trY, teY, trX, teX)
        return {
            "test_mae": mae,
            "train_val_inds": self.train_val_inds,
            "opt_reg": best["reg"],
            "opt_width": float(np.round(best["width"], 3)),
            "all_test_errors": np.asarray(pred) - np.asarray(teY),
            "test_inds": self.test_inds,
            "tss": len(self.train_val_inds),
            "bounds": vars(self.bounds),
            "n_dims": int(self.X.shape[1]),
        }

class Cross_Validation_Local(Cross_Validation):
    """
    Local-kernel CV variant using provided kernel builders.
    You must supply:
      - get_local_symmetric_kernels(X, Zs, width_values) -> List[Array] each (N x N)
      - get_local_kernel(X_train, X_eval, Z_train, Z_eval, width) -> Array (M x N)
    """
    def __init__(
        self,
        X: Array,
        Y: Array,
        nuclear_charges_list: Array,
        train_val_inds: Array,
        test_inds: Array,
        bounds: Dict,
        *,
        get_local_symmetric_kernels: Callable[[Array, Array, Sequence[float]], List[Array]],
        get_local_kernel: Callable[[Array, Array, Array, Array, float], Array],
        random_state: Optional[int] = None,
        solver: Callable[[Array, Array], Array] = stable_cholesky_solve,
        error_fn: Callable[[Array, Array], float] = rmse,
    ):
        super().__init__(X, Y, train_val_inds, test_inds, bounds, random_state, solver, error_fn)
        self.Z = nuclear_charges_list
        self.get_local_symmetric_kernels = get_local_symmetric_kernels
        self.get_local_kernel = get_local_kernel

    def pointer_k_fold_split(self, k: int) -> Tuple[List[Array], List[Array]]:
        """Split by pointers into the *train_val_inds* space (used with precomputed K)."""
        pointers = np.arange(self.train_val_inds.size)
        folds = np.array_split(self.rng.permutation(pointers), k)
        train_folds, val_folds = [], []
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i])
            train_folds.append(train_idx)
            val_folds.append(val_idx)
        return train_folds, val_folds

    def grid_search(self, num_folds: int) -> Dict[str, float]:
        widths = self._width_grid()
        regs = self._reg_grid()
        best_mae = np.inf
        best = {"width": None, "reg": None}

        idx = self.train_val_inds
        X_tv = self.X[idx]
        Z_tv = self.Z[idx]
        Y_tv = self.Y[idx]
        K_list = self.get_local_symmetric_kernels(X_tv, Z_tv, widths)

        tr_ptrs, vl_ptrs = self.pointer_k_fold_split(num_folds)

        for w_idx, w in enumerate(widths):
            K = K_list[w_idx]  # (N x N) symmetric
            for r in regs:
                fold_mae = []
                for trp, vlp in zip(tr_ptrs, vl_ptrs):
                    Ktt = K[np.ix_(trp, trp)]
                    Kvt = K[np.ix_(vlp, trp)]
                    y_tr = Y_tv[trp]; y_vl = Y_tv[vlp]
                    Ktt[np.diag_indices_from(Ktt)] += r
                    alpha = self.solver(Ktt, y_tr)
                    pred_vl = Kvt @ alpha
                    pred_tr = Ktt @ alpha
                    fold_mae.append(0.5 * (self.error_fn(y_tr, pred_tr) + self.error_fn(y_vl, pred_vl)))
                avg = float(np.mean(fold_mae))
                if avg < best_mae:
                    best_mae = avg
                    best = {"width": float(w), "reg": float(r)}
        return best

    def build(self, num_folds: int) -> Dict:
        best = self.grid_search(num_folds)
        idx = self.train_val_inds
        X_tr = self.X[idx]; Z_tr = self.Z[idx]; y_tr = self.Y[idx]
        X_te = self.X[self.test_inds]; Z_te = self.Z[self.test_inds]; y_te = self.Y[self.test_inds]
        w, r = best["width"], best["reg"]
        Ktt = self.get_local_kernel(X_tr, X_tr, Z_tr, Z_tr, w)
        Kqt = self.get_local_kernel(X_tr, X_te, Z_tr, Z_te, w)  # NOTE: train vs query order
        Ktt[np.diag_indices_from(Ktt)] += r
        alpha = self.solver(Ktt, y_tr)
        pred = Kqt @ alpha
        mae = float(np.mean(np.abs(y_te - pred)))
        return {
            "test_mae": mae,
            "train_val_inds": self.train_val_inds,
            "opt_reg": r,
            "opt_width": float(np.round(w, 3)),
            "all_test_errors": np.asarray(pred) - np.asarray(y_te),
            "test_inds": self.test_inds,
            "tss": len(self.train_val_inds),
            "bounds": vars(self.bounds),
            "n_dims": int(self.X.shape[1]),
        }

class CoreSetOptimizationFPS:
    """
    Orchestrates GGCS/FPS selection + kernel CV.
    """
    def __init__(
        self,
        X: Array,
        Y: Array,
        dZ_norm: Array,
        ls_D: Array,
        ls_inds: Array,
        css: int,
        us_inds: Array,
        bounds: Dict,
        grad_biases: Sequence[float],
        gb_start_frac: float,
        gb_stop_frac: float,
        nuc_charges: Optional[Array] = None,
        *,
        random_state: Optional[int] = None,
        # Optional deps for local mode:
        get_local_symmetric_kernels: Optional[
            Callable[[Array, Array, Sequence[float]], List[Array]]
        ] = None,
        get_local_kernel: Optional[
            Callable[[Array, Array, Array, Array, float], Array]
        ] = None,
    ):
        self.X = X
        self.Y = Y
        self.dZ_norm = dZ_norm
        self.ls_D = ls_D
        self.ls_inds = np.asarray(ls_inds, dtype=int)
        self.css = int(css)
        self.test_inds = np.asarray(us_inds, dtype=int)
        self.bounds = bounds
        self.grad_biases = list(grad_biases)
        self.gb_start_frac = float(gb_start_frac)
        self.gb_stop_frac = float(gb_stop_frac)
        self.nuc_charges = nuc_charges
        self.rng = ensure_rng(random_state)
        self.get_local_symmetric_kernels = get_local_symmetric_kernels
        self.get_local_kernel = get_local_kernel

    def _select_core(self, grad_bias: float) -> Array:
        return adaptive_fps_sweep(
            gradients=self.dZ_norm[self.ls_inds],
            D=self.ls_D,
            n_indices=self.css,
            gamma_start=-grad_bias * self.gb_start_frac,
            gamma_end=  grad_bias * self.gb_stop_frac,
            random_state=int(self.rng.integers(0, 2**32 - 1)),
        )

    def get_coreset(
        self,
        k_fun: Optional[Callable[[Array, Array, float], Array]],
        num_folds: int,
    ) -> Dict:
        best_mae = np.inf
        best_params = {"core_set_inds": None, "grad_bias": None, "width": None, "reg": None}
        for gb in self.grad_biases:
            core_local = self._select_core(gb)
            train_inds = self.ls_inds[core_local]
            val_inds = np.setdiff1d(self.ls_inds, train_inds, assume_unique=False)

            if self.nuc_charges is not None:
                if (self.get_local_symmetric_kernels is None) or (self.get_local_kernel is None):
                    raise ValueError("Local mode requires get_local_symmetric_kernels and get_local_kernel callables.")
                cv = Cross_Validation_Local(
                    self.X, self.Y, self.nuc_charges, train_inds, val_inds, self.bounds,
                    get_local_symmetric_kernels=self.get_local_symmetric_kernels,
                    get_local_kernel=self.get_local_kernel,
                    random_state=int(self.rng.integers(0, 2**32 - 1)),
                )
                results = cv.build(num_folds)
            else:
                if k_fun is None:
                    raise ValueError("Global mode requires k_fun.")
                cv = Cross_Validation(
                    self.X, self.Y, train_inds, val_inds, self.bounds,
                    random_state=int(self.rng.integers(0, 2**32 - 1)),
                )
                results = cv.build(k_fun, num_folds)

            if results["test_mae"] < best_mae:
                best_mae = results["test_mae"]
                best_params["core_set_inds"] = train_inds
                best_params["grad_bias"] = gb
                best_params["width"] = results["opt_width"]
                best_params["reg"] = results["opt_reg"]
        return best_params

    def evaluate(
        self,
        k_fun: Optional[Callable[[Array, Array, float], Array]],
        num_folds: int,
    ) -> Dict:
        best = self.get_coreset(k_fun, num_folds)
        if self.nuc_charges is not None:
            cv = Cross_Validation_Local(
                self.X, self.Y, self.nuc_charges, best["core_set_inds"], self.test_inds, self.bounds,
                get_local_symmetric_kernels=self.get_local_symmetric_kernels,
                get_local_kernel=self.get_local_kernel,
                random_state=int(self.rng.integers(0, 2**32 - 1)),
            )
            test_mae, pred = cv.evaluate(
                {"width": best["width"], "reg": best["reg"]},
                self.Y[best["core_set_inds"]], self.Y[self.test_inds],
                self.X[best["core_set_inds"]], self.X[self.test_inds],
                self.nuc_charges[best["core_set_inds"]], self.nuc_charges[self.test_inds],
            )  # type: ignore[arg-type]
        else:
            if k_fun is None:
                raise ValueError("Global mode requires k_fun.")
            cv = Cross_Validation(self.X, self.Y, best["core_set_inds"], self.test_inds, self.bounds)
            test_mae, pred = cv.evaluate(
                k_fun, {"width": best["width"], "reg": best["reg"]},
                self.Y[best["core_set_inds"]], self.Y[self.test_inds],
                self.X[best["core_set_inds"]], self.X[self.test_inds],
            )

        return {
            "all_test_errors": np.asarray(pred) - np.asarray(self.Y[self.test_inds]),
            "test_mae": float(test_mae),
            "test_inds": self.test_inds,
            "train_val_inds": best["core_set_inds"],
            "opt_grad_bias": best["grad_bias"],
            "opt_reg": float(best["reg"]),
            "opt_width": float(best["width"]),
            "lss": int(np.asarray(self.ls_inds).size),
            "css": int(self.css),
            "bounds": self.bounds,
            "n_dims": int(self.X.shape[1]),
        }
