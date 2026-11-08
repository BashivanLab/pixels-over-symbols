"""Per-task Dynamical Similarity Analysis (DSA).

For every system (one seed of a model) and task, the trajectories are projected onto their own
principal components, a HAVOK DMD operator is fit to the delay-embedded projection (``DSA.DMD``),
and the operators of every pair of systems are compared (:mod:`analysis.dsa.simdist`).
"""

import numpy as np
import torch
from DSA import DMD
from sklearn.decomposition import PCA

from analysis.dsa.simdist import compute_sim_matrix
from utils.helper import seed_everything


def per_system_pca(features, model_keys, n_components, random_state):
    """Project every system's trajectories onto its own top principal components, per task.

    Args:
        features: ``{model_key -> {seed -> {task_id -> (T, N, H) array}}}`` of states
            (time x trials x units).
        model_keys: Models to include.
        n_components: Number of components kept.
        random_state: Passed to ``sklearn.decomposition.PCA``.

    Returns:
        ``{"{model}_s{seed}" -> {task_id -> (N, T, n_components)}}``. Each PCA is fit on one
        system and task, with every time step of every trial as a sample.
    """
    reduced = dict()
    for model in model_keys:
        for seed, tasks in features[model].items():
            system = reduced[f'{model}_s{seed}'] = dict()
            for task_id, states in tasks.items():
                states = np.asarray(states).transpose(1, 0, 2)  # (N, T, H)
                N, T, H = states.shape
                X = states.reshape(-1, H)
                pca = PCA(n_components=n_components, random_state=random_state).fit(X)
                system[task_id] = pca.transform(X).reshape(N, T, n_components)
    return reduced


def fit_dmd(pca_features, n_delays, rank):
    """Fit a HAVOK DMD operator to every system and task.

    Args:
        pca_features: ``{system -> {task_id -> (N, T, n)}}`` from :func:`per_system_pca`.
        n_delays: Delay-embedding depth.
        rank: Number of right singular vectors of the Hankel matrix the operator acts on.

    Returns:
        ``{task_id -> {system -> (rank, rank) operator}}``.
    """
    operators = dict()
    for system, tasks in pca_features.items():
        for task_id, states in tasks.items():
            dmd = DMD(torch.tensor(states, dtype=torch.float32), n_delays=n_delays, rank=rank)
            dmd.fit()
            operators.setdefault(task_id, dict())[system] = dmd.A_v.cpu().numpy()
    return operators


def run_dsa_per_task(
        features,
        model_keys,
        n_pca_components,
        n_delays=2,
        rank=3,
        score_method='angular',
        iters=200,
        lr=5e-3,
        random_state=2026,
):
    """PCA, DMD and pairwise operator distances, separately for every task.

    Args:
        features: ``{model_key -> {seed -> {task_id -> (T, N, H) array}}}``.
        model_keys: Models to compare; every seed of each is one system.
        n_pca_components: PCA dimension.
        n_delays, rank: DMD settings (see :func:`fit_dmd`).
        score_method: ``'angular'``, ``'euclidean'`` or ``'wasserstein'``
            (see :func:`analysis.dsa.simdist.compute_sim_matrix`).
        iters, lr: Optimization of the orthogonal alignment.
        random_state: Seeds the PCA and torch's global RNG, which the alignment draws its
            initialization from.

    Returns:
        Tuple ``(distances, systems)``: ``{task_id -> (S, S) distance matrix}`` and the
        ``S`` system names ``"{model}_s{seed}"`` in row order.
    """
    seed_everything(random_state)
    pca_features = per_system_pca(features, model_keys, n_pca_components, random_state)
    systems = list(pca_features)
    operators = fit_dmd(pca_features, n_delays, rank)
    distances = {
        task_id: compute_sim_matrix(ops, systems, score_method=score_method, iters=iters, lr=lr)
        for task_id, ops in operators.items()
    }
    return distances, systems
