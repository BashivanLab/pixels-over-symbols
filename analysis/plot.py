"""Bar panels of the behavioral metrics.

One bar per model, in the given order, with its error bar, the model's marker above it and bracket
labels for its groups under the x-axis. The human value is a bar of its own for accuracy and a band
(mean +/- std) for rho(m,h) and kappa. Styles come from scripts/figures/styles.py.
save_bar_panels writes the panels of one slice of the results (all trials, a task family, or the
trials of one stimulus feature from feature_level_results).
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.helper import fisher_z_inverse, fisher_z_transform
from analysis.metrics.accuracy import accuracy_hierarchical
from analysis.metrics.consistency import consistency_hierarchical
from analysis.metrics.error_consistency import error_consistency_hierarchical

LINE_COLOR = '#333333'
LINE_STYLE = '--'
LINE_WIDTH = 1.0

DEFAULT_FIG_SIZE = (7, 6)


def _add_markers_to_bars(ax, model_order, model_style, values, yerr_high, offset=0.03):
    """Draw each model's marker just above its error bar (none on a zero-height bar)."""
    for i, model in enumerate(model_order):
        style = model_style.get(model, {})
        if 'marker' in style and not values[i] == 0:
            ax.plot(
                i, values[i] + yerr_high[i] + offset,
                marker=style['marker'], color=style.get('color', 'gray'),
                markersize=8, markeredgecolor='black', markeredgewidth=0.8,
                linestyle='None', zorder=5
            )


def _add_nested_group_brackets(ax, model_order, model_style, base_y=-0.1, level_gap=0.08):
    """Brackets under the x-axis spanning runs of bars with the same group label, one row per level."""
    max_depth = max(len(model_style.get(m, {}).get('group', [])) for m in model_order)

    for depth in range(max_depth):
        y_shift = base_y - (max_depth - 1 - depth) * level_gap

        labels = []
        for m in model_order:
            groups = model_style.get(m, {}).get('group', [])
            if depth == max_depth - 1:
                # the last row gives every model its innermost label
                label = groups[-1] if groups else 'Other'
            elif depth < len(groups) - 1:
                # outer rows only bracket models that have a deeper label
                label = groups[depth]
            else:
                label = 'Other'
            labels.append(label)

        spans = []
        current = labels[0]
        start = 0
        for i, g in enumerate(labels[1:], 1):
            if g != current:
                spans.append((current, start, i - 1))
                current = g
                start = i
        spans.append((current, start, len(labels) - 1))

        for name, start, end in spans:
            if name == 'Other':
                continue

            center = (start + end) / 2
            h = 0.03

            ax.plot(
                [start, start, end, end],
                [y_shift + h, y_shift, y_shift, y_shift + h],
                color='black', lw=1.2,
                transform=ax.get_xaxis_transform(),
                clip_on=False
            )

            ax.text(
                center, y_shift - 0.02,
                name,
                transform=ax.get_xaxis_transform(),
                ha='center', va='top',
                fontsize=10,
                clip_on=False
            )


def _models(task_results, model_order):
    return [m for m in model_order if m != 'human' and m in task_results.get('models', {})]


def _row(label, model_style, **values):
    style = model_style.get(label, {})
    return {'Agent': label, 'Color': style.get('color', 'gray'), 'Hatch': style.get('hatch', ''), **values}


def _human_band(ax, low, mean, high):
    if low == high:
        high = low + 0.01
    ax.axhspan(low, high, color=LINE_COLOR, alpha=0.1, zorder=0)
    ax.axhline(mean, color=LINE_COLOR, linestyle=LINE_STYLE, linewidth=LINE_WIDTH)


def _bars(ax, df, yerr, err_high, model_style, ylabel, ylim, yticks=True, **errorbar_kw):
    """Bars of df['value'] in row order, then error bars, markers, brackets and the axis labels."""
    agents = df['Agent'].tolist()
    sns.barplot(
        data=df, x='Agent', y='value', hue='Agent',
        palette=dict(zip(df['Agent'], df['Color'])),
        ax=ax, edgecolor='black', linewidth=0.8, dodge=False,
        order=agents, legend=False
    )
    bars = [p for p in ax.patches if isinstance(p, mpatches.Rectangle) and p.get_label() != '']
    for bar, hatch in zip(bars, df['Hatch']):
        bar.set_hatch(hatch)

    ax.errorbar(x=range(len(agents)), y=df['value'], yerr=yerr, fmt='none', capsize=3, **errorbar_kw)
    _add_markers_to_bars(ax, agents, model_style, df['value'].values, err_high)
    _add_nested_group_brackets(ax, agents, model_style, level_gap=0.12)

    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, fontsize=12)
    if yticks:
        ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlabel('')
    sns.despine(ax=ax)
    return ax


def plot_task_accuracy(task_results, model_order, model_style, ax=None, ylim=(0.0, 1.1)):
    """Accuracy, mean +/- std over seeds (or the CI where a summary has one); human as the last bar."""
    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    rows = []
    for label in _models(task_results, model_order) + ['human']:
        res = task_results.get('human', {}) if label == 'human' else task_results['models'][label]
        mean = res.get('mean', 0)
        if 'ci_lower' in res and 'ci_upper' in res:
            err_low, err_high = mean - res['ci_lower'], res['ci_upper'] - mean
        else:
            err_low = err_high = res.get('std', 0)
        rows.append(_row(label, model_style, value=mean, err_low=err_low, err_high=err_high))
    df = pd.DataFrame(rows).fillna(0)
    return _bars(ax, df, [df['err_low'], df['err_high']], df['err_high'].values, model_style, 'Accuracy', ylim,
                 c=LINE_COLOR, elinewidth=1.2)


def plot_task_corr(task_results, model_order, model_style, ax=None, ylim=(0.0, 1.1)):
    """rho(m,h), mean +/- std over split halves taken in Fisher z; human split-half reliability as a band."""
    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    rows = []
    for m in _models(task_results, model_order):
        r_mean = task_results['models'][m].get('mean', np.nan)
        z_std = task_results['models'][m].get('std', np.nan)
        z_mean = fisher_z_transform(r_mean)
        r_high = fisher_z_inverse(z_mean + z_std)
        r_low = fisher_z_inverse(z_mean - z_std)
        rows.append(_row(m, model_style, value=r_mean, err_low=r_mean - r_low, err_high=r_high - r_mean))

    human_res = task_results.get('human', None)
    if human_res is not None:
        r_h_mean = human_res.get('mean', np.nan)
        z_h_std = human_res.get('std', 0)
        z_h_mean = fisher_z_transform(r_h_mean)
        if not np.isnan(r_h_mean):
            _human_band(ax, fisher_z_inverse(z_h_mean - z_h_std), r_h_mean, fisher_z_inverse(z_h_mean + z_h_std))

    df = pd.DataFrame(rows).fillna(0)
    return _bars(ax, df, [df['err_low'].values, df['err_high'].values], df['err_high'].values, model_style,
                 r'$\rho (m, h)$', ylim, c='black', elinewidth=1.2)


def plot_error_consistency(task_results, model_order, model_style, ax=None, ylim=(-0.2, 0.5)):
    """Error consistency kappa, mean +/- std over subject-seed pairs; human-human kappa as a band."""
    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    rows = [_row(m, model_style, value=task_results['models'][m].get('mean', np.nan),
                 std=task_results['models'][m].get('std', np.nan))
            for m in _models(task_results, model_order)]

    human_res = task_results.get('human', None)
    if human_res is not None:
        h_mean = human_res.get('mean', np.nan)
        h_std = human_res.get('std', 0)
        if not np.isnan(h_mean):
            _human_band(ax, h_mean - h_std, h_mean, h_mean + h_std)

    df = pd.DataFrame(rows).fillna(0)
    return _bars(ax, df, df['std'], df['std'].values, model_style, r'$\kappa$', ylim, yticks=False, c='black')


def plot_rsa_alignment_summary(experiment_data, which, model_order, model_style, ax=None, ylim=(0.0, 1.1),
                               confidence_level=0.95):
    """Cross-task (which='task') or cross-stimulus ('seq') RSA alignment with humans: the Pearson r
    between the model's and the human correlation matrices, with its confidence interval."""
    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    rsa_data = experiment_data['rsa_results'][{'task': 'task_rsa_alignment', 'seq': 'seq_rsa_alignment'}[which]]
    rows = []
    for m in [m for m in model_order if m in rsa_data['models']]:
        res = rsa_data['models'][m]['mean']
        rho = res.statistic
        ci = res.confidence_interval(confidence_level=confidence_level)
        rows.append(_row(m, model_style, value=rho, err_low=rho - ci.low, err_high=ci.high - rho))
    df = pd.DataFrame(rows).fillna(0)
    return _bars(ax, df, [df['err_low'], df['err_high']], df['err_high'].values, model_style, r'$\rho$', ylim,
                 c='black')


def get_summary_data(results, key, model_names, mode='acc', agg='per_task'):
    """One slice of a hierarchical result: {'models': {model: summary}, 'human': summary}.

    The slice is results['overall'] or results[agg][key]. mode 'mh' picks each model's rho(m,h),
    anything else the summary as stored.
    """
    task_res = results.get(agg, {})
    if agg != 'overall':
        task_res = task_res.get(key, {})
    if not task_res:
        return {'human': None, 'models': {}}

    models_raw = task_res.get('models', {})
    models_data = {m: models_raw[m]['mh'] if mode == 'mh' else models_raw[m] for m in model_names if m in models_raw}
    return {'human': task_res.get('human'), 'models': models_data}


# metric -> bar function, results key, get_summary_data mode
BAR_PANELS = {
    'correlation': (plot_task_corr, 'cons', 'mh'),
    'accuracy': (plot_task_accuracy, 'acc', 'acc'),
    'error_consistency': (plot_error_consistency, 'ec', 'acc'),
}


def save_bar_panels(results, models, model_style, out_dir, panels, agg='overall', key='overall'):
    """Save one bar panel per {file stem: metric} into out_dir.

    results holds the hierarchical 'cons', 'acc' and 'ec' results; a panel shows their 'overall'
    slice, or the slice agg ('per_task_type', ...) and key.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, metric in panels.items():
        plot_func, result, mode = BAR_PANELS[metric]
        fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
        plot_func(get_summary_data(results[result], key, models, mode=mode, agg=agg), models, model_style, ax=ax)
        fig.savefig(out_dir / f'{stem}.png', bbox_inches='tight', dpi=600)
        plt.close(fig)


def feature_level_results(experiment_data):
    """Consistency, accuracy and error consistency of every model on the trials that compare each
    stimulus feature: {feature: {'cons': ..., 'acc': ..., 'ec': ...}}, each with an 'overall' slice."""
    all_model_conf = experiment_data['all_model_conf']
    all_model_acc = experiment_data['all_model_acc']
    human_responses = experiment_data['human_responses']
    task_metadata = experiment_data['new_task_metadata']
    n_models = len(next(iter(all_model_conf.values())).values())
    assert n_models == len(next(iter(all_model_acc.values())).values())

    results = {}
    for feat, task_indices in experiment_data['feat_index'].items():
        cons, _ = consistency_hierarchical(
            all_model_outputs=all_model_conf,
            human_responses=human_responses,
            n_pairs=100, task_metadata=task_metadata,
            model_splits=experiment_data['model_splits'], human_splits=experiment_data['human_splits'],
            task_indices=task_indices, aggs=('overall',)
        )
        acc, _ = accuracy_hierarchical(
            all_model_outputs=all_model_acc,
            human_responses=human_responses,
            n_pairs=100, task_metadata=task_metadata,
            task_indices=task_indices, aggs=('overall',)
        )
        ec, _ = error_consistency_hierarchical(
            all_model_acc=all_model_acc,
            human_responses=human_responses,
            n_pairs=100,
            n_models=n_models,
            task_metadata=task_metadata,
            task_indices=task_indices, aggs=('overall',)
        )
        results[feat] = {'cons': cons, 'acc': acc, 'ec': ec}
    return results
