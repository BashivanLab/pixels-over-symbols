"""Split a stack of DSA distance matrices into within-model and between-model distances."""

from itertools import combinations

import numpy as np


def compute_within_cross_3d(dsa_tensor, model_labels):
    """Split a stack of distance matrices into within-model and cross-model distances.

    Args:
        dsa_tensor: ``(T, M, M)`` stack of ``T`` symmetric distance matrices over the
            same ``M`` items (e.g. per-task DSA distances stacked over tasks).
        model_labels: Length-``M`` labels assigning each item to a model.

    Returns:
        Tuple ``(within, cross)``:
        - ``within``: ``{model -> (T, num_pairs)}`` upper-triangle distances among that
          model's items (``np.nan`` if it has fewer than two items).
        - ``cross``: ``{(model_a, model_b) -> (T, ka*kb)}`` all pairwise distances
          between the two models' items, with the model names in sorted order.
    """
    model_labels = np.array(model_labels)
    unique_models = np.unique(model_labels)
    T = dsa_tensor.shape[0]

    within = {}
    cross = {}

    for m in unique_models:
        idx = np.where(model_labels == m)[0]
        if len(idx) < 2:
            within[m] = np.nan
            continue
        block = dsa_tensor[:, idx][:, :, idx]  # (T, k, k)
        # each unordered pair once, without the zero diagonal
        rows, cols = np.triu_indices(len(idx), k=1)
        within[m] = block[:, rows, cols]  # (T, pairs)

    for m_a, m_b in combinations(unique_models, 2):
        idx_a = np.where(model_labels == m_a)[0]
        idx_b = np.where(model_labels == m_b)[0]
        block = dsa_tensor[:, idx_a][:, :, idx_b]  # (T, ka, kb)
        cross[(m_a, m_b)] = block.reshape(T, -1)

    return within, cross
