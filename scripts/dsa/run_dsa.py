"""DSA of the substitution experiments (Sec. 4.3 and the DSA figures of the appendix).

For every system (one seed of a model; the list is in analysis/dsa/systems.py) and each of the 12
MTurk tasks, the sequence-core hidden states h_t ('seq') and the projected inputs z_proj ('pre') are
projected onto the system's own top mu principal components (time steps as samples, no whitening),
a HAVOK operator with 2 delays and rank 3 is fit, and every pair of systems is compared with the
angular Procrustes distance (Adam, 200 iterations, learning rate 5e-3, seed 2026), for
mu = 2, 5, 10, 20, 50.

usage: python scripts/dsa/run_dsa.py

Reads the hidden states cached by the evaluation (computed from the checkpoints if missing) and
writes EVAL_DIR/dsa_distances.npz with
    labels           the system of each row and column (6 seeds per system)
    ds               the values of mu
    seq_pc<mu>       (12 tasks, systems x seeds, systems x seeds) distances on h_t
    pre_pc<mu>       the same on z_proj
    n_delays, rank
About 8 minutes on 16 CPU threads.
"""
import os

# PCA and DMD run on multithreaded BLAS/LAPACK, whose results change in the last bits with the
# thread count; the non-convex alignment carries that into some distances (up to 0.04 at 10 PCs).
# The paper's run used 16 threads.
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_var] = '16'

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.dsa.dsa_task import run_dsa_per_task  # noqa: E402
from analysis.dsa.systems import SYSTEMS, load_features  # noqa: E402
from utils.paths import EVAL_DIR  # noqa: E402

PCS = [2, 5, 10, 20, 50]
N_DELAYS, RANK = 2, 3
OUT = EVAL_DIR / 'dsa_distances.npz'


def main():
    torch.set_num_threads(16)
    print(f'{len(SYSTEMS)} systems, writing {OUT}', flush=True)
    save = {'ds': np.array(PCS), 'n_delays': N_DELAYS, 'rank': RANK}
    for tag in ('seq', 'pre'):
        features = load_features(SYSTEMS, tag)
        for pc in PCS:
            t0 = time.time()
            dist, rows = run_dsa_per_task(features, SYSTEMS, n_pca_components=pc, n_delays=N_DELAYS, rank=RANK)
            save[f'{tag}_pc{pc}'] = np.stack(list(dist.values()))  # (n_tasks, S, S)
            save['labels'] = np.array([row.rsplit('_s', 1)[0] for row in rows])
            print(f'[{tag}] {pc} components done in {time.time() - t0:.0f}s', flush=True)
    np.savez(OUT, **save)
    print('saved', OUT, flush=True)


if __name__ == '__main__':
    main()
