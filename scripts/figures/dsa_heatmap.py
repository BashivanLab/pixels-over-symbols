"""Pairwise DSA distances between the native and the substituted models: heatmap and MDS embedding.

Per recipient (the one-hot abstract LSTM at its final checkpoint, the natural ResNet-CLIP LSTM) the
six native seeds and the six seeds with the other model's final sequence core frozen in (theta).
Each matrix is the seed-by-seed angular DSA distance at 50 PCA components, averaged over the 12
MTurk tasks; the heatmap colour range spans its off-diagonal entries.

Reads the DSA results of scripts/dsa/run_dsa.py from data/evaluations and writes
figures/dsa/final/{heatmap,mds}_{seq,pre}_pc50.png (seq: hidden state h_t, pre: input-pathway output
z_proj,t).

    python scripts/figures/dsa_heatmap.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.dsa.plot_dsa import plot_distance_heatmap, plot_mds_clusters  # noqa: E402
from analysis.transplant_configs import ARMS  # noqa: E402
from utils.paths import EVAL_DIR, FIGURE_DIR  # noqa: E402

DSA_RESULTS = EVAL_DIR / 'dsa_distances.npz'
OUT = FIGURE_DIR / 'dsa' / 'final'
PC = 50

# systems in plot order, with the style plot_distance_heatmap / plot_mds_clusters read; 'group'
# labels the heatmap brackets
SYSTEMS = [
    (ARMS['reg_abs_final']['native'], {'color': '#e6550d', 'group': 'AB', 'marker': 'o', 'filled': True}),
    (ARMS['reg_abs_final']['theta'], {'color': '#e6550d', 'group': 'AB_SEQ', 'marker': '^', 'filled': True}),
    (ARMS['reg_res']['native'], {'color': '#2182bd', 'group': 'ResNet', 'marker': 'o', 'filled': True}),
    (ARMS['reg_res']['theta_finaldonor'], {'color': '#2182bd', 'group': 'ResNet_SEQ', 'marker': '^', 'filled': True}),
]


def main():
    z = np.load(DSA_RESULTS, allow_pickle=True)
    labels = list(z['labels'])
    style = dict(SYSTEMS)
    idx = [i for key in style for i, lab in enumerate(labels) if lab == key]
    sub = [labels[i] for i in idx]
    assert all(sub.count(k) == 6 for k in style), {k: sub.count(k) for k in style}
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in ('seq', 'pre'):
        mat = z[f'{tag}_pc{PC}'].mean(axis=0)[np.ix_(idx, idx)]
        off = mat[~np.eye(len(idx), dtype=bool)]
        plot_distance_heatmap(mat, sub, model_style=style, vmin=np.nanmin(off), vmax=np.nanmax(off),
                              save_path=OUT / f'heatmap_{tag}_pc{PC}.png')
        plot_mds_clusters(mat, sub, style, save_path=OUT / f'mds_{tag}_pc{PC}.png')
        plt.close('all')
    print('systems:', list(style), flush=True)
    print('figures written under', OUT)


if __name__ == '__main__':
    main()
