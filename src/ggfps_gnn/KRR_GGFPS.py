#!/usr/bin/env python

import numpy as np

from GGFPS import adaptive_ggfps_sweep
from qmllib.solvers import cho_solve
from qmllib.kernels import get_local_symmetric_kernels, get_local_kernel


def rmse(pred_labels, test_labels):
    """Root-mean-square error."""
    return np.sqrt(np.sum((pred_labels - test_labels) ** 2) / pred_labels.size)


def run_cross_validation_local(train_val_inds, test_inds, X, Y, nuclear_charges_list, bounds, num_folds):
    """Convenience wrapper for local-kernel CV."""
    cv = Cross_Validation_Local(X, Y, nuclear_charges_list, train_val_inds, test_inds, bounds)
    results = cv.build(num_folds)
    return results


class Cross_Validation_Local:
    """
    Cross-validation for local kernels using precomputed symmetric Gram lists.
    Uses pointer-based folds into train_val_inds to reuse precomputed K blocks.
    """

    def __init__(self, X, Y, nuclear_charges_list, train_val_inds, test_inds, bounds):
        self.X = X
        self.Y = Y
        self.nuclear_charges_list = nuclear_charges_list
        self.train_val_inds = np.asarray(train_val_inds)
        self.test_inds = np.asarray(test_inds)
        self.bounds = bounds

    def k_fold_split(self, k):
        """(Unused in build) Classic k-fold on indices; avoids mutating external arrays."""
        perm = np.random.permutation(self.train_val_inds)
        val_inds_folds = np.array_split(perm, k)
        train_inds_folds = [np.concatenate(val_inds_folds[:i] + val_inds_folds[i + 1:]) for i in range(k)]
        return train_inds_folds, val_inds_folds

    def pointer_k_fold_split(self, k):
        """Split by pointers 0..len(train_val_inds)-1 to index precomputed K efficiently."""
        pointers = np.arange(self.train_val_inds.size)
        perm = np.random.permutation(pointers)
        val_inds_folds = np.array_split(perm, k)
        train_inds_folds = [np.concatenate(val_inds_folds[:i] + val_inds_folds[i + 1:]) for i in range(k)]
        return train_inds_folds, val_inds_folds

    def split_X_Y_charges(self, train_inds, val_inds):
        """Helper for explicit splits (not used in pointer-based path)."""
        train_X, val_X, test_X = [self.X[inds] for inds in [train_inds, val_inds, self.test_inds]]
        train_Y, val_Y, test_Y = [self.Y[inds] for inds in [train_inds, val_inds, self.test_inds]]
        train_charges, val_charges, test_charges = [
            self.nuclear_charges_list[inds] for inds in [train_inds, val_inds, self.test_inds]
        ]
        return train_X, val_X, test_X, train_Y, val_Y, test_Y, train_charges, val_charges, test_charges

    def grid_search(self, train_pointer_inds, val_pointer_inds):
        """Grid search over width/reg using precomputed symmetric Gram matrices for widths."""
        search_density = self.bounds['search_density']
        reg_bounds = self.bounds['reg_bounds']
        width_bounds = self.bounds['width_bounds']

        if isinstance(width_bounds, tuple):
            width_values = list(np.logspace(np.log10(width_bounds[0]), np.log10(width_bounds[1]), search_density))
        elif isinstance(width_bounds, list):
            width_values = width_bounds
        else:
            raise ValueError("width_bounds must be tuple or list")

        if isinstance(reg_bounds, tuple):
            reg_values = list(np.logspace(np.log10(reg_bounds[0]), np.log10(reg_bounds[1]), 4))
        elif isinstance(reg_bounds, list):
            reg_values = reg_bounds
        else:
            raise ValueError("reg_bounds must be tuple or list")

        best_mae = np.inf
        best_params = {'width': None, 'reg': None}

        idx = self.train_val_inds
        train_val_K_list = get_local_symmetric_kernels(self.X[idx], self.nuclear_charges_list[idx], width_values)
        train_val_Y = self.Y[idx]

        for width_i, width in enumerate(width_values):
            Ksym = train_val_K_list[width_i]
            for reg in reg_values:
                mae_list = []
                for tp_inds, vp_inds in zip(train_pointer_inds, val_pointer_inds):
                    train_K = Ksym[tp_inds][:, tp_inds]
                    val_K = Ksym[vp_inds][:, tp_inds]
                    y_tr = train_val_Y[tp_inds]
                    y_vl = train_val_Y[vp_inds]
                    mae = self.validate_hyperparams(reg, train_K, val_K, y_tr, y_vl)
                    mae_list.append(mae)
                avg_mae = np.mean(mae_list)
                if avg_mae < best_mae:
                    best_mae = avg_mae
                    best_params['width'] = width
                    best_params['reg'] = reg
        return best_params

    def validate_hyperparams(self, reg, train_K, val_K, train_Y, val_Y):
        """Regularized KRR fit on train_K and validate on val_K."""
        train_K[np.diag_indices_from(train_K)] += reg
        alpha = cho_solve(train_K, train_Y)
        pred_val_Y = np.dot(val_K, alpha)
        pred_train_Y = np.dot(train_K, alpha)
        mae_train = rmse(train_Y, pred_train_Y)
        mae_val = rmse(val_Y, pred_val_Y)
        return (mae_train + mae_val) / 2

    def evaluate(self, best_params, train_Y, test_Y, train_X, test_X, train_Zs, test_Zs):
        """Fit on all train/val with best params; evaluate on held-out test."""
        width, reg = best_params['width'], best_params['reg']
        train_K = get_local_kernel(train_X, train_X, train_Zs, train_Zs, width)
        test_K = get_local_kernel(train_X, test_X, train_Zs, test_Zs, width)
        train_K[np.diag_indices_from(train_K)] += reg
        alpha = cho_solve(train_K, train_Y)
        pred_test_Y = np.dot(test_K, alpha)
        test_mae = np.mean(np.abs(test_Y - pred_test_Y))
        return test_mae, pred_test_Y

    def build(self, num_folds):
        """End-to-end: CV to pick width/reg, then final fit and test evaluation."""
        train_pointer_inds, val_pointer_inds = self.pointer_k_fold_split(num_folds)
        best_params = self.grid_search(train_pointer_inds, val_pointer_inds)

        final_train_X = self.X[self.train_val_inds]
        final_train_Y = self.Y[self.train_val_inds]
        final_train_Zs = self.nuclear_charges_list[self.train_val_inds]
        test_X = self.X[self.test_inds]
        test_Y = self.Y[self.test_inds]
        test_Zs = self.nuclear_charges_list[self.test_inds]

        final_mae, pred_test_Y = self.evaluate(
            best_params, final_train_Y, test_Y, final_train_X, test_X, final_train_Zs, test_Zs
        )

        results = {
            'test_mae': final_mae,
            'train_val_inds': self.train_val_inds,
            'opt_reg': best_params['reg'],
            'opt_width': np.round(best_params['width'], 3),
            'all_test_errors': np.array(pred_test_Y) - np.array(self.Y[self.test_inds]),
            'test_inds': self.test_inds,
            'tss': len(self.train_val_inds),
            'bounds': self.bounds,
            'n_dims': self.X.shape[1],
        }
        return results


class CoreSetOptimizationGGFPS:
    """
    Select a core set via gradient-guided FPS sweep over ls_inds,
    run kernel CV on the selected core (train) vs the remainder (val),
    then evaluate on held-out test (us_inds).
    """

    def __init__(self, X, Y, dZ_norm, ls_D, ls_inds, css, us_inds, bounds, grad_biases, gb_start_frac, gb_stop_frac, nuc_charges=None):
        self.X = X
        self.Y = Y
        self.dZ_norm = dZ_norm
        self.ls_D = ls_D
        self.ls_inds = np.asarray(ls_inds)
        self.css = css
        self.test_inds = np.asarray(us_inds)
        self.bounds = bounds
        self.grad_biases = grad_biases
        self.gb_start_frac = gb_start_frac
        self.gb_stop_frac = gb_stop_frac
        self.nuc_charges = nuc_charges

    def get_coreset(self, k_fun, num_folds):
        """Sweep grad_bias, build CV for each core, keep the best by validation MAE."""
        best_mae = np.inf
        best_params = {'core_set_inds': None, 'grad_bias': None, 'width': None, 'reg': None}

        for grad_bias in self.grad_biases:
            GGCS_inds = adaptive_ggfps_sweep(
                self.dZ_norm[self.ls_inds],
                self.ls_D,
                self.css,
                gamma_start=-1 * grad_bias * self.gb_start_frac,
                gamma_end=grad_bias * self.gb_stop_frac,
                random_state=None,
            )
            train_inds = self.ls_inds[GGCS_inds]
            val_inds = np.delete(self.ls_inds, GGCS_inds)

            if self.nuc_charges is not None:
                cv = Cross_Validation_Local(self.X, self.Y, self.nuc_charges, train_inds, val_inds, self.bounds)
                results = cv.build(num_folds)
            else:
                cv = Cross_Validation(self.X, self.Y, train_inds, val_inds, self.bounds)
                results = cv.build(k_fun, num_folds)

            if results['test_mae'] < best_mae:
                best_mae = results['test_mae']
                best_params['core_set_inds'] = train_inds
                best_params['grad_bias'] = grad_bias
                best_params['width'] = results['opt_width']
                best_params['reg'] = results['opt_reg']

        return best_params

    def evaluate(self, k_fun, num_folds):
        """Build best core set, then evaluate final model on held-out test."""
        best_params = self.get_coreset(k_fun, num_folds)

        if self.nuc_charges is not None:
            cv = Cross_Validation_Local(self.X, self.Y, self.nuc_charges, best_params['core_set_inds'], self.test_inds, self.bounds)
            test_mae, pred_test_Y = cv.evaluate(
                best_params,
                self.Y[best_params['core_set_inds']], self.Y[self.test_inds],
                self.X[best_params['core_set_inds']], self.X[self.test_inds],
                self.nuc_charges[best_params['core_set_inds']], self.nuc_charges[self.test_inds],
            )
        else:
            cv = Cross_Validation(self.X, self.Y, best_params['core_set_inds'], self.test_inds, self.bounds)
            test_mae, pred_test_Y = cv.evaluate(
                k_fun, best_params,
                self.Y[best_params['core_set_inds']], self.Y[self.test_inds],
                self.X[best_params['core_set_inds']], self.X[self.test_inds],
            )

        return {
            'all_test_errors': np.array(pred_test_Y) - np.array(self.Y[self.test_inds]),
            'test_mae': test_mae,
            'test_inds': self.test_inds,
            'train_val_inds': best_params['core_set_inds'],
            'opt_grad_bias': best_params['grad_bias'],
            'opt_reg': best_params['reg'],
            'opt_width': best_params['width'],
            'lss': np.array(self.ls_inds).size,
            'css': self.css,
            'bounds': self.bounds,
            'n_dims': self.X.shape[1],
        }


class Cross_Validation:
    """
    Cross-validation for global kernels (k_fun(Xa, Xb, width) -> Gram).
    """

    def __init__(self, X, Y, train_val_inds, test_inds, bounds):
        self.X = X
        self.Y = Y
        self.train_val_inds = np.asarray(train_val_inds)
        self.test_inds = np.asarray(test_inds)
        self.bounds = bounds

    def k_fold_split(self, k):
        """k-fold split that does NOT mutate external arrays."""
        perm = np.random.permutation(self.train_val_inds)
        val_inds_folds = np.array_split(perm, k)
        train_inds_folds = [np.concatenate(val_inds_folds[:i] + val_inds_folds[i + 1:]) for i in range(k)]
        return train_inds_folds, val_inds_folds

    def split_X_Y(self, train_inds, val_inds):
        train_X, val_X, test_X = [self.X[inds] for inds in [train_inds, val_inds, self.test_inds]]
        train_Y, val_Y, test_Y = [self.Y[inds] for inds in [train_inds, val_inds, self.test_inds]]
        return train_X, val_X, test_X, train_Y, val_Y, test_Y

    def grid_search(self, k_fun, train_inds_folds, val_inds_folds):
        """Grid search over width/reg for global kernel k_fun."""
        search_density = self.bounds['search_density']
        reg_bounds = self.bounds['reg_bounds']
        width_bounds = self.bounds['width_bounds']

        if isinstance(width_bounds, tuple):
            width_values = list(np.logspace(np.log10(width_bounds[0]), np.log10(width_bounds[1]), search_density))
        elif isinstance(width_bounds, list):
            width_values = width_bounds
        else:
            raise ValueError("width_bounds must be tuple or list")

        if isinstance(reg_bounds, tuple):
            reg_values = list(np.logspace(np.log10(reg_bounds[0]), np.log10(reg_bounds[1]), 4))
        elif isinstance(reg_bounds, list):
            reg_values = reg_bounds
        else:
            raise ValueError("reg_bounds must be tuple or list")

        best_mae = np.inf
        best_params = {'width': None, 'reg': None}

        for width in width_values:
            for reg in reg_values:
                mae_list = []
                for train_inds, val_inds in zip(train_inds_folds, val_inds_folds):
                    train_X, val_X, _, train_Y, val_Y, _ = self.split_X_Y(train_inds, val_inds)
                    mae = self.validate_hyperparams(k_fun, width, reg, train_Y, val_Y, train_X, val_X)
                    mae_list.append(mae)
                avg_mae = np.mean(mae_list)
                if avg_mae < best_mae:
                    best_mae = avg_mae
                    best_params['width'] = width
                    best_params['reg'] = reg
        return best_params

    def validate_hyperparams(self, k_fun, width, reg, train_Y, val_Y, train_X, val_X):
        train_K, val_K = k_fun(train_X, train_X, width), k_fun(val_X, train_X, width)
        train_K[np.diag_indices_from(train_K)] += reg
        alpha = cho_solve(train_K, train_Y)
        pred_val_Y = np.dot(val_K, alpha)
        pred_train_Y = np.dot(train_K, alpha)
        mae_train = rmse(train_Y, pred_train_Y)
        mae_val = rmse(val_Y, pred_val_Y)
        return (mae_train + mae_val) / 2

    def evaluate(self, k_fun, best_params, train_Y, test_Y, train_X, test_X):
        width, reg = best_params['width'], best_params['reg']
        train_K, test_K = k_fun(train_X, train_X, width), k_fun(test_X, train_X, width)
        train_K[np.diag_indices_from(train_K)] += reg
        alpha = cho_solve(train_K, train_Y)
        pred_test_Y = np.dot(test_K, alpha)
        test_mae = np.mean(np.abs(test_Y - pred_test_Y))
        return test_mae, pred_test_Y

    def build(self, k_fun, num_folds):
        train_inds_folds, val_inds_folds = self.k_fold_split(num_folds)
        best_params = self.grid_search(k_fun, train_inds_folds, val_inds_folds)

        final_train_X = self.X[self.train_val_inds]
        final_train_Y = self.Y[self.train_val_inds]
        test_X = self.X[self.test_inds]
        test_Y = self.Y[self.test_inds]

        final_mae, pred_test_Y = self.evaluate(k_fun, best_params, final_train_Y, test_Y, final_train_X, test_X)

        results = {
            'test_mae': final_mae,
            'train_val_inds': self.train_val_inds,
            'opt_reg': best_params['reg'],
            'opt_width': np.round(best_params['width'], 3),
            'all_test_errors': np.array(pred_test_Y) - np.array(self.Y[self.test_inds]),
            'test_inds': self.test_inds,
            'tss': len(self.train_val_inds),
            'bounds': self.bounds,
            'n_dims': self.X.shape[1],
        }
        return results


def run_cross_validation(train_val_inds, test_inds, X, Y, hyperparam_bounds, k_fun, num_folds):
    """Convenience wrapper for global-kernel CV."""
    cv = Cross_Validation(X, Y, train_val_inds, test_inds, hyperparam_bounds)
    results = cv.build(k_fun, num_folds)
    return results
