"""Heatmap and MDS plots of DSA distance matrices."""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import MDS


def _add_group_brackets_heatmap(ax, keys, model_style,
                                x_offset=-0.6, y_offset=-0.6,
                                text_offset=0.25):
    """Draw a labelled bracket on the left and bottom of a heatmap for each run of keys of one group.

    The group of a key is ``model_style[key]['group']`` (``'Other'`` when missing).
    """
    groups = []
    for k in keys:
        g = model_style.get(k, {}).get('group', ['Other'])
        groups.append(g[0] if isinstance(g, list) else g)

    # contiguous runs of the same group: (name, first index, last index)
    spans = []
    current = groups[0]
    start = 0
    for i, g in enumerate(groups):
        if g != current:
            spans.append((current, start, i - 1))
            current = g
            start = i
    spans.append((current, start, len(groups) - 1))

    # left brackets (rows)
    for name, start, end in spans:
        y0, y1 = start, end + 1
        ax.plot([x_offset, x_offset], [y0, y1],
                color='black', lw=1.5, clip_on=False)
        ax.plot([x_offset, x_offset + 0.2], [y0, y0],
                color='black', lw=1.5, clip_on=False)
        ax.plot([x_offset, x_offset + 0.2], [y1, y1],
                color='black', lw=1.5, clip_on=False)
        ax.text(x_offset - text_offset, (y0 + y1) / 2,
                name, ha='right', va='center',
                rotation=90, clip_on=False)

    # bottom brackets (columns)
    for name, start, end in spans:
        x0, x1 = start, end + 1
        ax.plot([x0, x1], [y_offset, y_offset],
                color='black', lw=1.5, clip_on=False)
        ax.plot([x0, x0], [y_offset, y_offset + 0.2],
                color='black', lw=1.5, clip_on=False)
        ax.plot([x1, x1], [y_offset, y_offset + 0.2],
                color='black', lw=1.5, clip_on=False)
        ax.text((x0 + x1) / 2, y_offset - text_offset,
                name, ha='center', va='bottom',
                clip_on=False)


def plot_distance_heatmap(
        dist_matrix: np.ndarray,
        keys: List[str],
        filtered_keys: Optional[List[str]] = None,
        key_order: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        x_ticks=None,
        y_ticks=None,
        vmin=None,
        vmax=None,
        model_style=None,
):
    """Heatmap of a distance matrix whose rows and columns are labelled by ``keys``.

    ``filtered_keys`` keeps only the rows and columns of those keys and ``key_order`` reorders them.
    With ``model_style`` the axes get group brackets (:func:`_add_group_brackets_heatmap`) instead
    of tick labels. The colour range defaults to the matrix's range.
    """
    if filtered_keys is not None:
        idx = [i for i, k in enumerate(keys) if k in filtered_keys]
        dist_matrix = dist_matrix[np.ix_(idx, idx)]
        keys = [keys[i] for i in idx]

    if key_order is not None:
        idx = [keys.index(k) for k in key_order if k in keys]
        dist_matrix = dist_matrix[np.ix_(idx, idx)]
        keys = [keys[i] for i in idx]

    fig, ax = plt.subplots(figsize=(12, 12))
    hm = sns.heatmap(
        dist_matrix,
        ax=ax,
        xticklabels=keys if x_ticks is None and model_style is None else False,
        yticklabels=keys if y_ticks is None and model_style is None else False,
        cmap="viridis_r",
        vmin=np.nanmin(dist_matrix) if vmin is None else vmin,
        vmax=np.nanmax(dist_matrix) if vmax is None else vmax,
        square=True,
        linewidths=0.1,
        cbar=False
    )
    fig.colorbar(
        hm.collections[0],
        ax=ax,
        fraction=0.03,
        pad=0.02
    )
    if model_style is not None:
        _add_group_brackets_heatmap(ax, keys, model_style)
    else:
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_mds_clusters(
        sim_matrix,
        model_keys,
        model_style,
        filtered_keys: Optional[List[str]] = None,
        key_order: Optional[List[str]] = None,
        save_path=None,
        random_state=2026,
        n_init=100,
):
    """2-D metric MDS of a precomputed distance matrix, one marker per key.

    ``model_style[key]`` gives the marker's ``color``, ``marker``, ``filled`` and ``alpha``;
    ``filtered_keys`` and ``key_order`` select and order the keys as in :func:`plot_distance_heatmap`.
    """
    if filtered_keys is not None:
        idx = [i for i, k in enumerate(model_keys) if k in filtered_keys]
        sim_matrix = sim_matrix[np.ix_(idx, idx)]
        model_keys = [model_keys[i] for i in idx]

    if key_order is not None:
        idx = [model_keys.index(k) for k in key_order if k in model_keys]
        sim_matrix = sim_matrix[np.ix_(idx, idx)]
        model_keys = [model_keys[i] for i in idx]

    coords = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=n_init,
    ).fit_transform(sim_matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, key in enumerate(model_keys):
        s = model_style[key]
        ax.scatter(
            coords[i, 0], coords[i, 1],
            c=s["color"] if s.get('filled', False) else "none",
            edgecolors=s["color"],
            marker=s["marker"],
            s=80,
            alpha=s.get("alpha", 0.9),
            linewidths=1,
        )

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
