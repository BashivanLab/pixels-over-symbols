"""Cross-task and cross-stimulus correlation matrices of responses, and their alignment (RSA).

The cross-task matrix correlates the responses of every pair of tasks over the 100 pairs, the
cross-stimulus matrix those of every pair of stimulus pairs over the 18 tasks; both correlate one half
of the participants (or seeds) with the other half, for random splits.
"""

import numpy as np
from scipy.stats import pearsonr
import seaborn as sns

N_SPLITS = 20


def get_cross_task_corr(data, n_splits=N_SPLITS):
    """Task x task correlation matrices, ``(n_tasks, n_tasks, n_splits)``, of ``data`` (n_subjects, n_tasks, n_stimuli).

    The halves are drawn from NumPy's global random state.
    """
    n_subjects, n_tasks, n_stimuli = data.shape
    matrix = np.zeros((n_tasks, n_tasks, n_splits))

    for s in range(n_splits):
        indices = np.random.permutation(n_subjects)
        h1_idx, h2_idx = indices[:n_subjects // 2], indices[n_subjects // 2:]
        # half means, stimuli x tasks
        h1_t = data[h1_idx].mean(axis=0).T
        h2_t = data[h2_idx].mean(axis=0).T

        # Pearson correlation of every task column of one half with every task column of the other
        h1_c = h1_t - h1_t.mean(axis=0, keepdims=True)
        h2_c = h2_t - h2_t.mean(axis=0, keepdims=True)
        h1_n = h1_c / (np.linalg.norm(h1_c, axis=0, keepdims=True) + 1e-8)
        h2_n = h2_c / (np.linalg.norm(h2_c, axis=0, keepdims=True) + 1e-8)
        matrix[:, :, s] = h1_n.T @ h2_n

    return matrix


def get_cross_sequence_corr(data, n_splits=N_SPLITS):
    """Stimulus x stimulus correlation matrices, ``(n_stimuli, n_stimuli, n_splits)``, of ``data`` (n_subjects, n_tasks, n_stimuli).

    The halves are drawn from NumPy's global random state.
    """
    n_subjects, n_tasks, n_stimuli = data.shape
    matrix = np.zeros((n_stimuli, n_stimuli, n_splits))

    for s in range(n_splits):
        indices = np.random.permutation(n_subjects)
        h1_idx, h2_idx = indices[:n_subjects // 2], indices[n_subjects // 2:]
        # half means, tasks x stimuli
        h1_mean = data[h1_idx].mean(axis=0)
        h2_mean = data[h2_idx].mean(axis=0)

        # Pearson correlation of every stimulus column of one half with every one of the other
        h1_c = h1_mean - h1_mean.mean(axis=0, keepdims=True)
        h2_c = h2_mean - h2_mean.mean(axis=0, keepdims=True)
        h1_n = h1_c / (np.linalg.norm(h1_c, axis=0, keepdims=True) + 1e-8)
        h2_n = h2_c / (np.linalg.norm(h2_c, axis=0, keepdims=True) + 1e-8)
        matrix[:, :, s] = h1_n.T @ h2_n

    return matrix


def compute_rsa_alignment(m1, m2):
    """Pearson correlation (scipy result) of the upper triangles of two matrices."""
    iu = np.triu_indices(m1.shape[0], k=1)
    return pearsonr(m1[iu], m2[iu])


def plot_matrix_comparison(
        ax,
        matrix,
        x_task_names=None,
        y_task_names=None,
        title=None,
        cmap='RdBu_r',
        vmin=-0.5,
        vmax=1.0,
        center=0.0,
        show_annotations=True,
        plot_x_ticks=True,
        plot_y_ticks=True,
        cbar=False,
        cbar_ax=None,
        cbar_kws=None,
        linewidths=0.5
):
    """Heatmap of a correlation (or difference) matrix, with cell values unless ``show_annotations`` is False."""
    im = sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        square=True,
        linewidths=linewidths,
        linecolor='white',
        annot=False,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws={'fraction': 0.045, 'pad': 0.02} if cbar_kws is None else cbar_kws
    )

    N = matrix.shape[0]

    if show_annotations:
        for i in range(N):
            for j in range(N):
                val = matrix[i, j]

                # white text on the dark ends of the colour map
                norm_val = (val - vmin) / (vmax - vmin + 1e-8)
                color = 'white' if norm_val < 0.3 or norm_val > 0.7 else 'black'

                ax.text(
                    j + 0.5, i + 0.5,
                    f"{val:.2f}",
                    ha='center', va='center',
                    fontsize=7,
                    color=color
                )

    if plot_x_ticks:
        ax.set_xticks(np.arange(N) + 0.5)
    else:
        ax.set_xticks([])

    if plot_y_ticks:
        ax.set_yticks(np.arange(N) + 0.5)
    else:
        ax.set_yticks([])

    if x_task_names is not None:
        ax.set_xticklabels(
            x_task_names,
            rotation=45,
            ha='right',
            fontsize=8
        )

    if y_task_names is not None:
        ax.set_yticklabels(
            y_task_names,
            rotation=0,
            fontsize=8
        )

    if not plot_x_ticks and not plot_y_ticks:
        ax.tick_params(length=0)

    if title is not None:
        ax.set_title(title, fontsize=13, pad=12)

    for spine in ax.spines.values():
        spine.set_visible(False)

    return im
