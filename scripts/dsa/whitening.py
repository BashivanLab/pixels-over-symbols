"""What whitening the PCA would change in the DSA (appendix, sensitivity to the DSA settings).

HAVOK fits the operator on the top-rank singular vectors of the Hankel matrix, which are orthonormal
whether or not the PCA components were whitened, so whitening only changes which rank-3 subspace of
the delay-embedded trajectories is kept. For every system of scripts/dsa/run_dsa.py and task, this
measures how much of the unwhitened delay-embedded trajectory (the same components, variance-weighted)
that subspace explains, with and without whitening, at 2 delays and rank 3.

usage: python scripts/dsa/whitening.py
Prints the median and 5th percentile over systems, seeds and tasks. A few minutes on a CPU.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.dsa.systems import SYSTEMS, load_features  # noqa: E402

N_DELAYS, RANK = 2, 3
PCS = [5, 10, 20, 50]


def hankel(x):
    """(N, T, n) trajectories -> (N * (T - d + 1), d * n) delay embedding, as in DSA.DMD."""
    T = x.shape[1]
    lagged = [x[:, i:T - N_DELAYS + 1 + i] for i in range(N_DELAYS)]
    return np.concatenate(lagged, axis=2).reshape(-1, N_DELAYS * x.shape[2])


def explained(signal, basis):
    """Share of ``signal`` explained by least squares on the columns of ``basis``."""
    coef, *_ = np.linalg.lstsq(basis, signal, rcond=None)
    return 1 - ((signal - basis @ coef) ** 2).sum() / (signal ** 2).sum()


def main():
    for tag in ('seq', 'pre'):
        share = {(pcs, whiten): [] for pcs in PCS for whiten in (False, True)}
        for model, seeds in load_features(SYSTEMS, tag).items():
            for tasks in seeds.values():
                for states in tasks.values():
                    x = np.asarray(states, dtype=np.float64).transpose(1, 0, 2)  # (N, T, H)
                    N, T, H = x.shape
                    X = x.reshape(-1, H)
                    for pcs, whiten in share:
                        pca = PCA(n_components=pcs, whiten=whiten, random_state=2026).fit(X)
                        z = pca.transform(X).reshape(N, T, pcs)
                        unwhitened = z * np.sqrt(pca.explained_variance_) if whiten else z
                        kept = np.linalg.svd(hankel(z), full_matrices=False)[0][:, :RANK]
                        share[pcs, whiten].append(explained(hankel(unwhitened), kept))
        print(f"== {'h_t' if tag == 'seq' else 'z_proj'}: share of the unwhitened delay-embedded trajectory "
              f'explained by the rank-{RANK} HAVOK subspace')
        for pcs in PCS:
            a, b = np.array(share[pcs, False]), np.array(share[pcs, True])
            print(f'  {pcs:2d} PCs: unwhitened median {np.median(a):.3f} (5th pct {np.percentile(a, 5):.3f}),'
                  f' whitened median {np.median(b):.3f} (5th pct {np.percentile(b, 5):.3f})')


if __name__ == '__main__':
    main()
