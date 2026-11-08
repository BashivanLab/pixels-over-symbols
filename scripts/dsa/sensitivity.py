"""DSA under other settings, for the sensitivity analysis in the appendix (tab:dsa_sensitivity).

Each configuration changes one setting of the default (10 PCA components, no whitening, 2 delays,
rank 3, angular distance, Adam with 200 iterations at learning rate 5e-3, seed 2026) and recomputes
the distances between the systems of scripts/dsa/run_dsa.py on h_t and z_proj.
scripts/dsa/sensitivity_report.py tests the comparisons of Sec. 4.3 on them.

usage: python scripts/dsa/sensitivity.py [config ...]      (default: all 19 configurations)

Writes EVAL_DIR/dsa_sensitivity/<config>.npz with labels, seq and pre ((12 tasks, S, S) float32
distances, as in run_dsa.py) and config. One to seven minutes per configuration on 12 CPU threads,
about an hour for all of them.
"""
import os

# The paper's sensitivity runs used 12 threads (run_dsa.py explains why the thread count matters), so
# 'base' matches them and not run_dsa.py at 10 components (differences up to 0.04).
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_var] = '12'

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.dsa.dsa_task import run_dsa_per_task  # noqa: E402
from analysis.dsa.systems import SYSTEMS, load_features  # noqa: E402
from utils.paths import EVAL_DIR  # noqa: E402

DEFAULT = dict(n_pca_components=10, n_delays=2, rank=3, score_method='angular', iters=200, lr=5e-3,
               random_state=2026)
CONFIGS = {
    'base': {},
    'pc2': {'n_pca_components': 2}, 'pc5': {'n_pca_components': 5},
    'pc20': {'n_pca_components': 20}, 'pc50': {'n_pca_components': 50},
    'delays1': {'n_delays': 1}, 'delays3': {'n_delays': 3},
    'rank2': {'rank': 2}, 'rank4': {'rank': 4},
    'euclidean': {'score_method': 'euclidean'}, 'wasserstein': {'score_method': 'wasserstein'},
    'it500': {'iters': 500}, 'it1000': {'iters': 1000},
    'it1000_lr1e-3': {'iters': 1000, 'lr': 1e-3}, 'it1000_lr1e-2': {'iters': 1000, 'lr': 1e-2},
    'seed0': {'random_state': 0}, 'seed7': {'random_state': 7},
    'seed42': {'random_state': 42}, 'seed2025': {'random_state': 2025},
}
OUT_DIR = EVAL_DIR / 'dsa_sensitivity'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('configs', nargs='*', help=f'any of {", ".join(CONFIGS)}')
    names = parser.parse_args().configs or list(CONFIGS)
    if set(names) - set(CONFIGS):
        parser.error(f'unknown configurations {sorted(set(names) - set(CONFIGS))}')
    torch.set_num_threads(12)
    OUT_DIR.mkdir(exist_ok=True)

    features = {tag: load_features(SYSTEMS, tag) for tag in ('seq', 'pre')}
    # with d delays a T-frame trial has T - d one-step transitions, which bounds the rank, as does the
    # size d * PCs of the delay embedding (the DMD would silently clip a larger rank)
    n_frames = min(np.asarray(x).shape[0] for f in features.values() for model in f.values()
                   for tasks in model.values() for x in tasks.values())
    for name in names:
        cfg = {**DEFAULT, **CONFIGS[name]}
        d, r, n_pc = cfg['n_delays'], cfg['rank'], cfg['n_pca_components']
        assert r <= n_frames - d and r <= d * n_pc, f'{name}: rank {r} too high for {d} delays, {n_pc} PCs'
        save = {'config': np.array(repr(cfg))}
        for tag in ('seq', 'pre'):
            t0 = time.time()
            dist, rows = run_dsa_per_task(features[tag], SYSTEMS, **cfg)
            save[tag] = np.stack(list(dist.values())).astype(np.float32)  # (n_tasks, S, S)
            save['labels'] = np.array([row.rsplit('_s', 1)[0] for row in rows])
            print(f'[{name}] {tag} done in {time.time() - t0:.0f}s', flush=True)
        np.savez(OUT_DIR / f'{name}.npz', **save)
    print('saved under', OUT_DIR, flush=True)


if __name__ == '__main__':
    main()
