"""Error consistency (Geirhos et al., NeurIPS 2020): Cohen's kappa of two responders' trial-by-trial correctness."""

import itertools
from typing import Dict, Union

import numpy as np


def compute_error_consistency(
        correct_i: Union[np.ndarray, list],
        correct_j: Union[np.ndarray, list]
) -> Dict[str, float]:
    """Error consistency of two observers on the same trials.

    "Beyond accuracy: quantifying trial-by-trial behaviour of CNNs and humans by measuring error
    consistency" (Geirhos et al., NeurIPS 2020).

    Args:
        correct_i, correct_j: correctness (bool or 0/1) of observers i and j on the same N trials.

    Returns:
        ``accuracy_i``, ``accuracy_j``, the observed and the expected (by chance) fraction of trials
        with equal correctness ``c_obs`` and ``c_exp``, and ``kappa`` = (c_obs - c_exp) / (1 - c_exp).
    """
    correct_i = np.asarray(correct_i, dtype=bool)
    correct_j = np.asarray(correct_j, dtype=bool)

    if len(correct_i) != len(correct_j):
        raise ValueError("The two arrays must have the same length (evaluate on the exact same stimuli).")

    n = len(correct_i)
    if n == 0:
        return {'accuracy_i': np.nan, 'accuracy_j': np.nan, 'c_obs': np.nan, 'c_exp': np.nan, 'kappa': np.nan}

    p_i = np.mean(correct_i)
    p_j = np.mean(correct_j)

    equal_responses = (correct_i == correct_j)
    e_ij = np.sum(equal_responses)
    c_obs = e_ij / n

    c_exp = (p_i * p_j) + ((1.0 - p_i) * (1.0 - p_j))

    if c_exp == 1.0:
        # both at 0% or both at 100% accuracy: kappa is 1 if they agree everywhere, else 0
        kappa = 1.0 if c_obs == 1.0 else 0.0
    else:
        kappa = (c_obs - c_exp) / (1.0 - c_exp)

    return {
        'accuracy_i': float(p_i),
        'accuracy_j': float(p_j),
        'c_obs': float(c_obs),
        'c_exp': float(c_exp),
        'kappa': float(kappa)
    }


def error_consistency_hierarchical(
        all_model_acc,
        human_responses,
        n_pairs,
        n_models,
        task_metadata,
        metric='kappa',
        task_indices=None,
        aggs=('per_task', 'per_task_type', 'overall'),
        feature_indices=None,
        acc_draws=None,
):
    """Error consistency of every model with every participant, and between participants, per aggregation level.

    A model's value is the mean kappa over all (participant, seed) pairs; the human value is the mean
    over all pairs of participants. ``acc_draws`` holds the per-draw correctness of noisy models,
    ``{key: {seed_id: (n_draws, n_trials)}}``: their ``all_model_acc`` entries are fractions of correct
    draws, so kappa is computed per draw and averaged over draws. Output layout as
    ``analysis.metrics.consistency.consistency_hierarchical``, with mean and std per model.
    """
    task_ids = sorted(task_metadata.keys())
    if task_indices is None:
        task_indices = {t: np.arange(i * n_pairs, (i + 1) * n_pairs) for i, t in enumerate(task_ids)}

    task_type_map = {}
    for t in task_ids:
        task_type_map.setdefault(task_metadata[t]['type'], []).append(t)

    results = {a: {} for a in aggs}
    samples = {a: {} for a in aggs}

    N_SUBJECTS = human_responses.shape[0]

    def compute_slice(idx):
        idx = np.array(idx).astype(int)
        m_results = {}
        m_samples = {}

        h_data = human_responses[:, idx]

        hh_vals = []
        for s1, s2 in itertools.combinations(range(N_SUBJECTS), 2):
            ec = compute_error_consistency(h_data[s1], h_data[s2])
            hh_vals.append(ec[metric])

        h_summary = {
            'mean': float(np.nanmean(hh_vals)),
            'std': float(np.nanstd(hh_vals))
        }

        for m, m_acc in all_model_acc.items():
            vals = []

            for h_idx in range(N_SUBJECTS):
                for seed in range(n_models):
                    m_key = f"{m}_{seed}"
                    if m_key not in m_acc:
                        continue

                    draws = (acc_draws or {}).get(m, {}).get(m_key)
                    if draws is not None:
                        vals.append(np.nanmean([
                            compute_error_consistency(h_data[h_idx], d[idx])[metric] for d in draws]))
                        continue
                    ec = compute_error_consistency(
                        h_data[h_idx],
                        m_acc[m_key][idx]
                    )
                    vals.append(ec[metric])

            vals = np.array(vals)

            m_results[m] = {
                'mean': float(np.nanmean(vals)),
                'std': float(np.nanstd(vals))
            }
            m_samples[m] = vals

        return {'models': m_results, 'human': h_summary}, {
            'models': m_samples,
            'human': np.array(hh_vals)
        }

    if 'per_task' in aggs:
        for t in task_ids:
            if t in task_indices:
                results['per_task'][t], samples['per_task'][t] = compute_slice(task_indices[t])

    if 'per_task_type' in aggs:
        for tt, tasks in task_type_map.items():
            idx = np.concatenate([task_indices[t] for t in tasks if t in task_indices])
            results['per_task_type'][tt], samples['per_task_type'][tt] = compute_slice(idx)

    if 'per_feature' in aggs:
        assert feature_indices is not None, "'per_feature' aggregation requires feature_indices"
        for f, idx in feature_indices.items():
            results['per_feature'][f], samples['per_feature'][f] = compute_slice(idx)

    if 'overall' in aggs:
        idx = np.concatenate([task_indices[t] for t in task_ids if t in task_indices])
        results['overall'], samples['overall'] = compute_slice(idx)

    return results, samples
