"""The pipeline's metrics as one long table.

:func:`summarize_pipeline` flattens the accuracy, consistency and error-consistency results of
:func:`analysis.experiment_pipeline.run_experiment_pipeline` into one DataFrame. The results are nested
``result[agg][group]['models'][model] -> stats`` (``overall`` has no group level). Consistency stats
carry ``mh`` (rho(m, h)), ``mm`` (rho(m, m)) and ``hc`` (noise-corrected) parts, and their top-level
``mean``/``std`` are those of ``hc``; the ``mh_*`` and ``mm_*`` columns are filled for consistency only.
"""

import numpy as np
import pandas as pd

METRIC_SOURCES = {
    'consistency': 'cons_result',
    'accuracy': 'acc_result',
    'error_consistency': 'ec_result',
}
AGGS = ('overall', 'per_feature', 'per_task_type', 'per_task')


def _rows_from_group(stats_by_model, human_stats, metric, agg, group):
    """Rows of one ``{'models': ..., 'human': ...}`` block."""
    entries = list(stats_by_model.items())
    if human_stats is not None:
        entries.append(('human', human_stats))
    rows = []
    for model, s in entries:
        row = {
            'metric': metric, 'agg_level': agg, 'group': group, 'model': model,
            'mean': s.get('mean', np.nan),
            'std': s.get('std', np.nan),
            'n_splits': s.get('n_splits', np.nan),
        }
        for part in ('mh', 'mm'):
            sub = s.get(part)
            row[f'{part}_mean'] = sub['mean'] if sub else np.nan
            row[f'{part}_std'] = sub['std'] if sub else np.nan
        rows.append(row)
    return rows


def summarize_pipeline(data, groups=None, metrics=None, aggs=AGGS,
                       include_random=True, include_human=True):
    """Flatten the pipeline's hierarchical metric dicts into one long DataFrame.

    Args:
        data: The dict returned by ``run_experiment_pipeline``.
        groups: Optional ``{family_name: [model_key, ...]}`` that fills the
            ``family`` column. A ``<key>_rand`` twin gets its model's family;
            keys not listed get an empty family.
        metrics: Subset of ``METRIC_SOURCES`` keys; default all three.
        aggs: Aggregation levels to include; default all four.
        include_random: Keep the ``*_rand`` random-init twins.
        include_human: Keep the per-group human rows (``model == 'human'``);
            for consistency this is human split-half reliability, not a
            model-vs-human score.

    Returns:
        DataFrame, one row per (metric, agg_level, group, model), with columns
        ``metric, agg_level, group, task_name, task_type, model, family, is_random,
        base_model, mean, std, n_splits, mh_mean, mh_std, mm_mean, mm_std``.
    """
    metrics = list(metrics or METRIC_SOURCES)
    task_meta = data.get('new_task_metadata', {})

    rows = []
    for metric in metrics:
        result = data[METRIC_SOURCES[metric]]
        for agg in aggs:
            if agg not in result:
                continue
            block = result[agg]
            if agg == 'overall':
                items = [('overall', block)]
            else:
                items = list(block.items())
            for group, entry in items:
                rows.extend(_rows_from_group(
                    entry['models'],
                    entry.get('human') if include_human else None,
                    metric, agg, group,
                ))

    df = pd.DataFrame(rows)

    df['is_random'] = df['model'].str.endswith('_rand')
    df['base_model'] = df['model'].str.replace(r'_rand$', '', regex=True)

    family = {}
    for name, keys in (groups or {}).items():
        for k in keys:
            family[k] = name
    df['family'] = df['base_model'].map(family).fillna('')
    df.loc[df['model'] == 'human', 'family'] = 'Human'

    # per_task groups are the renumbered task ids; carry their names/types along
    is_task = df['agg_level'] == 'per_task'
    df['task_name'] = ''
    df['task_type'] = ''
    df.loc[is_task, 'task_name'] = df.loc[is_task, 'group'].map(
        lambda t: task_meta.get(t, {}).get('name', ''))
    df.loc[is_task, 'task_type'] = df.loc[is_task, 'group'].map(
        lambda t: task_meta.get(t, {}).get('type', ''))

    if not include_random:
        df = df[~df['is_random']]

    cols = ['metric', 'agg_level', 'group', 'task_name', 'task_type', 'model', 'family',
            'is_random', 'base_model', 'mean', 'std', 'n_splits',
            'mh_mean', 'mh_std', 'mm_mean', 'mm_std']
    return df[cols].reset_index(drop=True)
