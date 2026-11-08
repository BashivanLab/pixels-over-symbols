"""Accuracy of models and participants on the MTurk trials, per task, task type, compared feature and overall."""

from typing import Dict

import numpy as np

from analysis.helper import summarize_std


def compute_model_accuracy(model_outputs: Dict, idx):
    """Per-seed accuracies on the trials `idx`, and their mean."""
    idx = np.array(idx).astype(int)
    accs = [vec[idx].mean() for vec in model_outputs.values()]
    return accs, float(np.mean(accs))


def compute_human_accuracy(human_responses, idx):
    """Per-participant accuracies on the trials `idx`, and their mean."""
    idx = np.array(idx).astype(int)
    accs = human_responses[:, idx].mean(axis=1)
    return accs.tolist(), float(np.mean(accs))


def accuracy_hierarchical(
        all_model_outputs, human_responses, n_pairs, task_metadata,
        task_indices=None, aggs=('per_task', 'per_task_type', 'overall'),
        feature_indices=None,
):
    """Accuracy per aggregation level: ``{agg: {group: {'models': {key: stats}, 'human': stats}}}``
    (``overall`` without the group level); stats are mean and std over seeds or participants.
    Also returns the per-seed / per-participant values in the same layout.
    """
    task_ids = sorted(task_metadata.keys())
    if task_indices is None:
        task_indices = {t: np.arange(i * n_pairs, (i + 1) * n_pairs) for i, t in enumerate(task_ids)}

    task_type_map = {}
    for t in task_ids:
        task_type_map.setdefault(task_metadata[t]['type'], []).append(t)

    results = {a: {} for a in aggs}
    samples = {a: {} for a in aggs}

    def process_slice(idx):
        m_samples = {}
        m_results = {}
        for m, m_out in all_model_outputs.items():
            s, _ = compute_model_accuracy(m_out, idx)
            m_samples[m] = s
            m_results[m] = summarize_std(s, is_correlation=False)

        h_s, _ = compute_human_accuracy(human_responses, idx)
        return ({'models': m_results, 'human': summarize_std(h_s, is_correlation=False)},
                {'models': m_samples, 'human': h_s})

    if 'per_task' in aggs:
        for t in task_ids:
            results['per_task'][t], samples['per_task'][t] = process_slice(task_indices[t])

    if 'per_task_type' in aggs:
        for tt, tasks in task_type_map.items():
            idx = np.concatenate([task_indices[t] for t in tasks if t in task_indices])
            results['per_task_type'][tt], samples['per_task_type'][tt] = process_slice(idx)

    if 'per_feature' in aggs:
        assert feature_indices is not None, "'per_feature' aggregation requires feature_indices"
        for f, idx in feature_indices.items():
            results['per_feature'][f], samples['per_feature'][f] = process_slice(np.asarray(idx))

    if 'overall' in aggs:
        idx = np.concatenate([task_indices[t] for t in task_ids if t in task_indices])
        results['overall'], samples['overall'] = process_slice(idx)

    return results, samples
