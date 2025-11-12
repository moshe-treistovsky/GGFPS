# #!/usr/bin/env python
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import numpy as np
import pandas as pd


from qmllib.kernels import distance
from qmllib.representations import generate_fchl19
from src.ggfps_gnn.KRR_GGFPS import CoreSetOptimizationGGFPS


def gen_flat_fchl(nuccs, coords):
    reps = []
    for ind, nucc in enumerate(nuccs):
        fchl = generate_fchl19(nucc, coords[ind], gradients=False, pad=nucc.size)
        flattened_array = np.ravel(fchl, order='C')
        reps.append(flattened_array)
    return np.array(reps)

def gen_fchl(nuccs, coords):
    reps = []
    for ind, nucc in enumerate(nuccs):
        fchl = generate_fchl19(nucc, coords[ind], gradients=False, pad=nucc.size)
        reps.append(fchl)
    return np.array(reps)

parser = argparse.ArgumentParser(description='FPS for Ethanol Surface')
parser.add_argument('lss', type=int, help='25, 50, 100, 250, 500')
parser.add_argument('shuffle_ind', type=int, help='bootstrap number')
parser.add_argument('mol_path', type=str, help='input molecule trajectory file path')
parser.add_argument('mol', type=str, help='input molecule')
parser.add_argument('gb_start_frac', type=float, help='grad bias start frac')
parser.add_argument('gb_stop_frac', type=float, help='grad bias stop frac')
parser.add_argument('tss_l', type=int, nargs='+', help='training set sizes list')
args = parser.parse_args()

# Use numpy to load the .npz file
data = np.load(args.mol_path, allow_pickle=True)

energies = data['E_train'].flatten()
forces = data['F_train']
geometries = data['R_train']
nuc_charges = data['z'].flatten()
nuccs = np.array([nuc_charges for _ in range(energies.size)])
force_norms_per_atom = np.linalg.norm(forces, axis=2)
forces_norms = np.sum(force_norms_per_atom, axis=1)


bounds = {'width_bounds': [8, 16, 32],
        'reg_bounds': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
        'search_density': None
        }

X =  gen_fchl(nuccs, geometries)

X_flat = gen_flat_fchl(nuccs, geometries)

all_inds = np.arange(energies.size)

lss = args.lss

tss_l = args.tss_l

grad_biases = np.linspace(0, 2, 10)#

all_ggfps_cvs = []

for i in np.arange(1):#
    ls_inds = np.random.choice(all_inds, lss, replace=False)  # Labeled set indices
    us_inds = np.delete(all_inds, ls_inds)  # Unlabeled set indices

    D = distance.l2_distance(X_flat[ls_inds], X_flat[ls_inds])

    for tss in tss_l:

        core_set_opt = CoreSetOptimizationGGFPS(np.copy(X), np.copy(energies), np.copy(forces_norms), np.copy(D), ls_inds, tss, us_inds, bounds, grad_biases, args.gb_start_frac, args.gb_stop_frac, np.copy(nuccs))
        GGFPS_cv = core_set_opt.evaluate(None, 5, test=False)
        GGFPS_cv['iteration'] = i

        all_ggfps_cvs.append(GGFPS_cv)

        print(f'ggfps test mae: {GGFPS_cv['test_mae']}, opt_reg:{GGFPS_cv['opt_reg']}, opt width:{GGFPS_cv['opt_width']}')


ggfps_df = pd.DataFrame(all_ggfps_cvs)

# Save to pickle
with open(f'{args.mol}_GGFPS_lss{args.lss}_ind{args.shuffle_ind}tss{tss_l[-1]}startfrac{args.gb_start_frac}stopfrac{args.gb_stop_frac}.pkl', 'wb') as f:
    pd.to_pickle(ggfps_df, f)