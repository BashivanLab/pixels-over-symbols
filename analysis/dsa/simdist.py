"""Pairwise distances between DMD operators.

The angular and Euclidean distances are a batched version of the Procrustes analysis over vector
fields of ``SimilarityTransformDist`` in the DSA package (Ostrow et al., 2023;
https://github.com/mitchellostrow/DSA, MIT License, Copyright (c) 2023 Mitchell Ostrow). As there,
each pair gets an orthogonal ``C = Cayley(Skew(I))``, where ``Skew`` is a bias-free 3-layer tanh MLP
followed by skew-symmetrization, optimized with Adam and refit once with the first two coordinates
of ``B`` swapped, keeping the fit with the lower loss. Here all pairs are optimized together.
The Wasserstein distance between eigenvalue spectra is ``SimilarityTransformDist``'s own.
"""

import numpy as np
import torch
from DSA.simdist import SimilarityTransformDist


def cayley_map(X):
    """Map a batch of skew-symmetric matrices ``(P, n, n)`` to orthogonal ones, ``(I + X)^-1 (I - X)``."""
    P, n, _ = X.shape
    eye = torch.eye(n).expand(P, n, n)
    return torch.linalg.solve(eye + X, eye - X)


def batched_skew_mlp(orig, W1, W2, W3):
    """DSA's ``Skew`` parametrization, applied to every pair.

    Args:
        orig: ``(P, n, n)`` base parameter of each pair.
        W1, W2, W3: ``(P, n, n)`` weights of each pair's MLP. ``nn.Linear(n, n, bias=False)``
            computes ``X @ W^T``, hence the transposes.

    Returns:
        ``(P, n, n)`` skew-symmetric matrices.
    """
    Y = torch.tanh(torch.bmm(orig, W1.transpose(1, 2)))
    Y = torch.tanh(torch.bmm(Y, W2.transpose(1, 2)))
    Y = torch.bmm(Y, W3.transpose(1, 2))
    return Y - Y.transpose(1, 2)


def optimize_C(A_batch, B_batch, iters, lr):
    """Fit an orthogonal ``C`` minimizing ``||A - C B C^T||_F^2`` for every pair.

    The parameters start as in DSA: the base matrix at the identity and each MLP weight drawn
    like ``nn.Linear(n, n, bias=False)``'s, from U(-1/sqrt(n), 1/sqrt(n)), with torch's global
    RNG. The draws for all pairs come in one block, so a pair's initialization depends on its
    position in the batch.

    Args:
        A_batch, B_batch: ``(P, n, n)`` operators.
        iters, lr: Adam iterations and learning rate.

    Returns:
        Tuple ``(C, loss)``: ``(P, n, n)`` orthogonal matrices and the ``(P,)`` final losses.
    """
    P, n, _ = A_batch.shape
    orig = torch.eye(n).expand(P, n, n).clone().requires_grad_(True)
    bound = 1.0 / (n ** 0.5)
    W1 = torch.empty(P, n, n).uniform_(-bound, bound).requires_grad_(True)
    W2 = torch.empty(P, n, n).uniform_(-bound, bound).requires_grad_(True)
    W3 = torch.empty(P, n, n).uniform_(-bound, bound).requires_grad_(True)
    optimizer = torch.optim.Adam([orig, W1, W2, W3], lr=lr)

    for _ in range(iters):
        optimizer.zero_grad()
        C = cayley_map(batched_skew_mlp(orig, W1, W2, W3))
        pred = torch.bmm(C, torch.bmm(B_batch, C.transpose(1, 2)))
        # summed over pairs, each pair's parameters get the gradient of that pair's loss alone
        ((A_batch - pred) ** 2).sum(dim=(1, 2)).sum().backward()
        optimizer.step()

    C = cayley_map(batched_skew_mlp(orig, W1, W2, W3)).detach()
    with torch.no_grad():
        pred = torch.bmm(C, torch.bmm(B_batch, C.transpose(1, 2)))
        loss = ((A_batch - pred) ** 2).sum(dim=(1, 2))
    return C, loss


def fit_similarity_batch_cayley(A_batch, B_batch, iters, lr):
    """Best orthogonal alignment ``C B C^T ~ A`` of every pair of unit-norm operators.

    Args:
        A_batch, B_batch: ``(P, n, n)`` operators.
        iters, lr: Adam iterations and learning rate.

    Returns:
        ``(P, n, n)`` orthogonal matrices ``C``.
    """
    A_batch = A_batch / torch.linalg.norm(A_batch, dim=(1, 2), keepdim=True)
    B_batch = B_batch / torch.linalg.norm(B_batch, dim=(1, 2), keepdim=True)
    P, n, _ = A_batch.shape
    C1, loss1 = optimize_C(A_batch, B_batch, iters, lr)

    # second start from B with its first two coordinates swapped; C2 @ perm undoes the swap
    perm = torch.eye(n)
    if n > 1:
        perm[[0, 1], :] = perm[[1, 0], :]
    perm = perm.unsqueeze(0).expand(P, -1, -1)
    C2, loss2 = optimize_C(A_batch, torch.bmm(perm, torch.bmm(B_batch, perm.transpose(1, 2))), iters, lr)
    C2 = torch.bmm(C2, perm)

    return torch.where((loss2 < loss1).view(-1, 1, 1), C2, C1)


def compute_sim_matrix_batched(A_matrices, keys, iters, lr, score_method):
    """Angular or Euclidean distance between every pair of operators, all pairs fit at once.

    Args:
        A_matrices: ``{key -> (n, n) operator}``.
        keys: The ``N`` keys to compare, in row order.
        iters, lr: Adam iterations and learning rate.
        score_method: ``'angular'``, ``arccos(<A, C B C^T> / (|A| |B|))``, or ``'euclidean'``,
            ``|A - C B C^T|`` on the operators as given.

    Returns:
        Symmetric ``(N, N)`` distance matrix with a zero diagonal.
    """
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
    A_batch = torch.stack([torch.tensor(A_matrices[a], dtype=torch.float32) for a, _ in pairs])
    B_batch = torch.stack([torch.tensor(A_matrices[b], dtype=torch.float32) for _, b in pairs])
    C = fit_similarity_batch_cayley(A_batch, B_batch, iters, lr)
    pred = torch.bmm(C, torch.bmm(B_batch, C.transpose(1, 2)))
    if score_method == 'angular':
        num = torch.einsum('bij,bij->b', A_batch, pred)
        den = torch.norm(A_batch, dim=(1, 2)) * torch.norm(B_batch, dim=(1, 2))
        # the clamp keeps rounding past +-1 out of arccos
        scores = torch.arccos(torch.clamp(num / den, -1.0, 1.0))
    elif score_method == 'euclidean':
        scores = torch.norm(A_batch - pred, dim=(1, 2))
    else:
        raise ValueError(f'unknown score_method {score_method!r}')

    n = len(keys)
    dist = np.zeros((n, n))
    dist[np.triu_indices(n, k=1)] = scores.numpy()
    return dist + dist.T


def compute_sim_matrix_wasserstein(A_matrices, keys):
    """Wasserstein distance between the eigenvalue spectra of every pair of operators.

    Args:
        A_matrices: ``{key -> (n, n) operator}``.
        keys: The ``N`` keys to compare, in row order.

    Returns:
        Symmetric ``(N, N)`` distance matrix with a zero diagonal.
    """
    metric = SimilarityTransformDist(score_method='wasserstein')
    n = len(keys)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = metric.fit_score(
                torch.tensor(A_matrices[keys[i]], dtype=torch.float32),
                torch.tensor(A_matrices[keys[j]], dtype=torch.float32),
            )
    return dist


def compute_sim_matrix(A_matrices, keys, score_method='angular', iters=200, lr=5e-3):
    """Pairwise DSA distances between DMD operators.

    Args:
        A_matrices: ``{key -> (n, n) operator}``, one per system.
        keys: The ``N`` keys to compare, in row order.
        score_method: ``'angular'`` or ``'euclidean'`` (after optimal orthogonal alignment), or
            ``'wasserstein'`` (between eigenvalue spectra, no alignment).
        iters, lr: Adam iterations and learning rate of the alignment.

    Returns:
        Symmetric ``(N, N)`` distance matrix with a zero diagonal.
    """
    if score_method == 'wasserstein':
        return compute_sim_matrix_wasserstein(A_matrices, keys)
    return compute_sim_matrix_batched(A_matrices, keys, iters, lr, score_method)
