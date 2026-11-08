"""Trial-by-trial consistency of models with humans.

For each split of the model seeds and of the participants into halves, the half-mean responses give
the model-human correlation rho(m, h) (Fisher mean of the four cross-half correlations), the model and
human split-half reliabilities rho(m, m) and rho(h, h), and the noise-corrected
rho(m, h) / sqrt(rho(m, m) rho(h, h)).
"""

import numpy as np
from scipy.stats import pearsonr

from analysis.helper import fisher_z_mean, summarize_std


def compute_consistency(
        rho_mh: float,
        rho_mm: float,
        rho_hh: float,
        internal_cutoff=0.1,
) -> float:
    """Noise-corrected consistency rho_mh / sqrt(rho_mm * rho_hh), capped at 1.

    NaN when either reliability is at most `internal_cutoff`, 0 when rho_mh <= 0.
    """
    if rho_mm <= internal_cutoff or rho_hh <= internal_cutoff:
        return np.nan
    elif rho_mh <= 0:
        return 0
    consistency = rho_mh / np.sqrt(rho_mm * rho_hh)
    return min(1.0, consistency)


def get_consistency(
        model_outputs,
        human_responses,
        model_splits,
        human_splits,
        idx=None,
        internal_cutoff=0.1,
):
    """rho(m, h), rho(m, m), rho(h, h) and the noise-corrected consistency ('hc') of every split, on the trials `idx`."""
    assert len(model_splits) == len(human_splits)

    if idx is not None:
        idx = np.array(idx).astype(int)
        model_data = np.array([v[idx] for v in model_outputs.values()])
        human_data = human_responses[:, idx]
    else:
        model_data = np.array([v for v in model_outputs.values()])
        human_data = human_responses

    res = {
        'mh': [], 'mm': [], 'hh': [], 'hc': [],
        'model_responses': [], 'human_responses': []
    }

    for (h_idx_a, h_idx_b), (m_idx_a, m_idx_b) in zip(human_splits, model_splits):
        h_a = human_data[h_idx_a].mean(axis=0)
        h_b = human_data[h_idx_b].mean(axis=0)
        hh_i = pearsonr(h_a, h_b)[0]

        m_a = model_data[m_idx_a].mean(axis=0)
        m_b = model_data[m_idx_b].mean(axis=0)
        mm_i = pearsonr(m_a, m_b)[0]

        cross_rs = [
            pearsonr(m_a, h_a)[0],
            pearsonr(m_a, h_b)[0],
            pearsonr(m_b, h_a)[0],
            pearsonr(m_b, h_b)[0]
        ]
        mh_i = fisher_z_mean(cross_rs)

        hc_i = compute_consistency(
            mh_i, mm_i, hh_i,
            internal_cutoff=internal_cutoff
        )

        res['hh'].append(hh_i)
        res['mm'].append(mm_i)
        res['mh'].append(mh_i)
        res['hc'].append(hc_i)
        res['human_responses'].append((h_a, h_b))
        res['model_responses'].append((m_a, m_b))
    return res


def consistency_hierarchical(
        all_model_outputs, human_responses, n_pairs, task_metadata,
        model_splits, human_splits, internal_cutoff=0.1,
        task_indices=None, aggs=('per_task', 'per_task_type', 'overall'),
        feature_indices=None,
):
    """Consistency of every model with the participants, per aggregation level.

    Args:
        all_model_outputs: ``{key: {'<key>_<seed>': (n_trials,)}}`` response vectors (confidences).
        human_responses: ``(n_participants, n_trials)`` correctness.
        n_pairs: trials per task.
        task_metadata: ``{task id: {'name', 'type'}}`` of the tasks, in response-vector order.
        model_splits, human_splits: split halves of the seeds and of the participants, paired by position.
        internal_cutoff: reliability below which the noise-corrected value is NaN.
        task_indices: trial indices of each task; by default consecutive blocks of `n_pairs`.
        aggs: any of 'per_task', 'per_task_type', 'per_feature', 'overall'.
        feature_indices: ``{feature: trial indices}`` (first output of
            ``analysis.helper.get_feature_indices``), needed for 'per_feature'.

    Returns:
        ``(results, samples)``. ``results[agg][group]`` (no group level for 'overall') holds
        ``'models': {key: stats}`` and ``'human': stats``. A model's stats hold 'mh', 'mm' and 'hc'
        (each mean and std over splits; correlations averaged in Fisher z), with the top-level
        'mean'/'std' those of 'hc'; the human stats are those of rho(h, h). ``samples`` holds the
        per-split 'hc' values of the models and rho(h, h) values of the participants.
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
        m_results, m_samples = {}, {}
        split_data = None
        for m, m_out in all_model_outputs.items():
            split_data = get_consistency(
                m_out, human_responses, model_splits, human_splits,
                idx=idx, internal_cutoff=internal_cutoff
            )
            m_results[m] = summarize_std(split_data['hc'], is_correlation=False)
            m_results[m].update({
                'hc': summarize_std(split_data['hc'], is_correlation=False),
                'mh': summarize_std(split_data['mh']),
                'mm': summarize_std(split_data['mm'])
            })
            m_samples[m] = split_data['hc']

        h_summary = summarize_std(split_data['hh']) if split_data else None
        h_samples = split_data['hh'] if split_data else []
        return {'models': m_results, 'human': h_summary}, {'models': m_samples, 'human': h_samples}

    if 'per_task' in aggs:
        for t in task_ids:
            if t in task_indices:
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
