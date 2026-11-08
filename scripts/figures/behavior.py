"""Behavioral figures of the paper, drawn from the cached evaluation results in data/evaluations.

No model is run: load() stops if a cached result is missing. Each set writes its panels under
figures/<set>/ (the substitution sets under figures/transplant/<final|acc_matched|dm>/); correlation
is rho(m,h), error_consistency is kappa:
  seq_type          abstract (80%-matched) and natural models with each sequence core, and the
                    VLMs: correlation, accuracy and error_consistency over all trials, the same
                    per task family (task_level/) and stimulus feature (feature_level/), the human
                    cross-task and cross-stimulus correlation matrices (RSA/), and the LSTMs'
                    cross-task matrices and their differences from the human one (RSA_ct_panels/)
  ab_match_levels   the abstract cores at the 75%-matched, 80%-matched and final checkpoints next
                    to the natural LSTM
  vision_type       LSTM and Transformer cores on eight vision encoders
  seq_size          lines of rho(m,h) and accuracy against the hidden size (64-1024)
  encoder_size      the same against the size of ConvNeXt and EVA02-CLIP encoders, LSTM core
  noise_dose        dimensionality-matched LSTM against the noise level s
  metric_alignment  rho(m,h) against accuracy, kappa and RSA across models (rho_*.png), and the
                    linear separability of each encoder's features against the metrics of the LSTM
                    trained on it (<feature>_<metric>.png)
  transplant_final, transplant_acc_matched, transplant_dm
                    substitutions between the abstract and the natural (ResNet-CLIP) LSTM, with
                    the one-hot abstract models at final or 80%-matched checkpoints, or with the
                    dimensionality-matched abstract model; RSA bars for the final set

    PYTHONHASHSEED=0 python scripts/figures/behavior.py [set ...]
"""
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import styles as S  # noqa: E402
from analysis.cross_task import plot_matrix_comparison  # noqa: E402
from analysis.experiment_pipeline import cache_stem, run_experiment_pipeline  # noqa: E402
from analysis.factory import AB_CONFIG_DICT, BACKBONES, DIM_MATCH_CONFIG_DICT, NAT_CONFIG_DICT  # noqa: E402
from analysis.helper import fisher_z_inverse, fisher_z_transform  # noqa: E402
from analysis.plot import (  # noqa: E402
    DEFAULT_FIG_SIZE, LINE_COLOR, feature_level_results, get_summary_data, plot_rsa_alignment_summary, save_bar_panels,
)
from analysis.transplant_configs import AB_CONFIGS, ARMS, DM_CONFIGS, NAT_CONFIGS  # noqa: E402
from utils.const import VLM_MODELS  # noqa: E402
from utils.paths import EVAL_DIR, FIGURE_DIR  # noqa: E402

ALL_AB = {**AB_CONFIG_DICT, **AB_CONFIGS}
ALL_NAT = {**NAT_CONFIG_DICT, **NAT_CONFIGS}
ALL_DM = {**DIM_MATCH_CONFIG_DICT, **DM_CONFIGS}


def load(name, keys, vlm=None, random=False):
    """Evaluation pipeline on the cached results, with every key's six seeds; prints accuracy and rho(m,h).

    random: also load each model's random-initialization twin (`<key>_rand`).
    """
    if os.environ.get('PYTHONHASHSEED') != '0':
        sys.exit('Run with PYTHONHASHSEED=0 (scripts/figures/make_all.sh sets it): the pipeline visits the '
                 'models in set order and draws their RSA split halves from one random stream, so the RSA '
                 'results depend on the hash seed.')
    ab = [k for k in keys if k in ALL_AB]
    dm = [k for k in keys if k in ALL_DM]
    nat = [k for k in keys if k in ALL_NAT]
    assert len(ab) + len(dm) + len(nat) == len(keys), f'unknown keys: {set(keys) - set(ab + dm + nat)}'
    # the cache files the pipeline would otherwise compute
    files = ['human_responses.npy', 'feature_indices.npy', 'pair_keys.npy']
    files += [f'{k}.npy' for k in vlm or []]
    for k in keys:
        cfg = {**ALL_AB, **ALL_NAT, **ALL_DM}[k]
        stems = [cache_stem(k, cfg)] + ([f'{k}_rand'] if random else [])
        for stem in stems:
            files += [f'{stem}_conf.npy', f'{stem}_acc.npy']
            if cfg.get('noise_scale', 0) > 0:
                files.append(f'{stem}_acc_draws.npy')
    missing = [f for f in files if not (EVAL_DIR / f).is_file()]
    if missing:
        raise FileNotFoundError(f'not in {EVAL_DIR}: {missing}')

    np.random.seed(0)
    data = run_experiment_pipeline(
        name, ab_model_keys=ab, nat_model_keys=nat, dim_match_model_keys=dm, vlm_model_keys=vlm,
        custom_ab_configs=AB_CONFIGS, custom_nat_configs=NAT_CONFIGS, custom_dim_match_configs=DM_CONFIGS,
        n_splits=20, device='cpu', load_random=random)
    short = {k: len(data['all_model_conf'][k]) for k in keys if len(data['all_model_conf'][k]) != 6}
    assert not short, f'keys without 6 seeds: {short}'

    acc, cons = data['acc_result']['overall'], data['cons_result']['overall']
    print(f"\n[{name}] {'model':40s} {'acc':>6s} {'rho(m,h)':>9s}", flush=True)
    for k in keys + (vlm or []):
        print(f"[{name}] {k:40s} {acc['models'][k]['mean']:6.3f} {cons['models'][k]['mh']['mean']:9.3f}", flush=True)
    print(f"[{name}] {'human':40s} {acc['human']['mean']:6.3f}", flush=True)
    return data


RHO_ACC = {'correlation': 'correlation', 'accuracy': 'accuracy'}  # file stem -> metric of a bar panel


def overall(data):
    """The pipeline's hierarchical results in the layout save_bar_panels reads."""
    return {'cons': data['cons_result'], 'acc': data['acc_result'], 'ec': data['ec_result']}


def marker(m):
    return Line2D([], [], marker=m, linestyle='', color='black', markeredgecolor='black', markersize=9)


def save_legend(handles, labels, path):
    """A legend as its own file (the panels carry none), one entry per row."""
    fig = plt.figure(figsize=(2.4, 0.4 * len(handles) + 0.2))
    fig.legend(handles, labels, loc='center', frameon=False, fontsize=10, labelspacing=1.0)
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)


def rho_interval(res):
    """rho(m,h) mean and mean -/+ the split-half std taken in Fisher z."""
    z = fisher_z_transform(res['mean'])
    return res['mean'], fisher_z_inverse(z - res['std']), fisher_z_inverse(z + res['std'])


def std_interval(res):
    return res['mean'], res['mean'] - res['std'], res['mean'] + res['std']


# the two panels of the line plots: stem, result, get_summary_data mode, error bars, y label
LINE_PANELS = [('correlation', 'cons_result', 'mh', rho_interval, r'$\rho (m, h)$'),
               ('accuracy', 'acc_result', 'acc', std_interval, 'Accuracy')]


def human_band(ax, mean, low, high):
    ax.axhspan(low, max(high, low + 0.01), color=LINE_COLOR, alpha=0.1, zorder=0)
    ax.axhline(mean, color=LINE_COLOR, linestyle='--', linewidth=1.0, label='Human', zorder=1)


def seq_type():
    keys = S.SEQ_TYPE_KEYS
    models = keys + VLM_MODELS
    styles = S.SEQ_TYPE_STYLES
    data = load('seq_type', keys, vlm=VLM_MODELS)
    out = FIGURE_DIR / 'seq_type'
    save_bar_panels(overall(data), models, styles, out, {**RHO_ACC, 'error_consistency': 'error_consistency'})
    for family in sorted({m['type'] for m in data['new_task_metadata'].values()}):
        save_bar_panels(overall(data), models, styles, out / 'task_level' / f'per_task_type_{family}', RHO_ACC,
                        agg='per_task_type', key=family)
    for feat, results in feature_level_results(data).items():
        save_bar_panels(results, models, styles, out / 'feature_level' / feat, RHO_ACC)
    # marker shapes only; the brackets of the panels name Abstract / Natural / VLM
    save_legend([marker(S.CORE_MARKER[c]) for c in S.CORES] + [marker('D')],
                [c.replace('DECODER', 'Transformer') for c in S.CORES] + ['VLM'], out / 'legend.png')
    rsa_matrices(data, keys, out)


def rsa_matrices(data, keys, out):
    """The human cross-task and cross-stimulus correlation matrices (RSA/human_baseline_{ct,cs}.png), and
    in RSA_ct_panels/ the cross-task matrix C_task of humans and of the two LSTMs, and the LSTMs'
    differences from the human matrix on the symmetric scale +-L shared by all models in keys (L = the
    largest absolute difference rounded up to 0.1) with its colour bar."""
    rsa = data['rsa_results']
    meta = data['new_task_metadata']
    task_names = [meta[i + 1]['name'] for i in range(len(meta))]
    (out / 'RSA').mkdir(parents=True, exist_ok=True)
    (out / 'RSA_ct_panels').mkdir(parents=True, exist_ok=True)

    def save(fig, path):
        fig.savefig(path, bbox_inches='tight', dpi=600)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    plot_matrix_comparison(ax, rsa['human_cross_task'], x_task_names=task_names, y_task_names=task_names, cbar=True)
    save(fig, out / 'RSA' / 'human_baseline_ct.png')
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_matrix_comparison(ax, rsa['human_cross_seq'], show_annotations=False, plot_x_ticks=False, plot_y_ticks=False,
                           cbar=True, cbar_kws={'fraction': 0.045, 'pad': 0.02}, linewidths=0)
    save(fig, out / 'RSA' / 'human_baseline_cs.png')

    human = np.asarray(rsa['human_cross_task'], dtype=float)
    model = {k: np.asarray(rsa['model_cross_task'][k], dtype=float) for k in keys}
    lim = max(np.nanmax(np.abs(model[k] - human)) for k in keys)
    L = math.ceil(lim * 10 - 1e-9) / 10
    print(f'[seq_type] max |model - human| C_task = {lim:.3f}, difference scale +-{L}', flush=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_matrix_comparison(ax, human, x_task_names=task_names, y_task_names=task_names, vmin=-1, vmax=1, cbar=True)
    save(fig, out / 'RSA_ct_panels' / 'human_ct.png')
    for k in ['ResNet_CLIP_LSTM_NOMTURK', 'AB_LSTM_NOMTURK_U80']:
        fig, ax = plt.subplots(figsize=(7, 7))
        plot_matrix_comparison(ax, model[k], vmin=-1, vmax=1, show_annotations=False,
                               plot_x_ticks=False, plot_y_ticks=False, cbar=False)
        save(fig, out / 'RSA_ct_panels' / f'{k}_ct.png')
        fig, ax = plt.subplots(figsize=(7, 7))
        plot_matrix_comparison(ax, model[k] - human, cmap='PiYG', vmin=-L, vmax=L, show_annotations=False,
                               plot_x_ticks=False, plot_y_ticks=False, cbar=False)
        save(fig, out / 'RSA_ct_panels' / f'{k}_ct_minus_human.png')
    # plot_matrix_comparison centres its colour map at 0, which on a symmetric range is a plain linear map
    fig, ax = plt.subplots(figsize=(0.4, 6))
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-L, L), cmap='PiYG'), cax=ax,
                 label=r'$C_{\mathrm{task}}$ (model $-$ human)')
    save(fig, out / 'RSA_ct_panels' / 'colorbar_ct_diff.png')


def ab_match_levels():
    # the legend of the paper's figure is seq_type/legend.png
    data = load('ab_match_levels', S.AB_MATCH_LEVEL_KEYS)
    save_bar_panels(overall(data), S.AB_MATCH_LEVEL_KEYS, S.AB_MATCH_LEVEL_STYLES, FIGURE_DIR / 'ab_match_levels',
                    RHO_ACC)


def vision_type():
    keys = S.VISION_TYPE_KEYS
    data = load('vision_type', keys)
    out = FIGURE_DIR / 'vision_type'
    save_bar_panels(overall(data), keys, S.VISION_TYPE_STYLES, out, RHO_ACC)
    vision_legend(out / 'legend.png')


def vision_legend(path):
    """Legend of the vision_type bars: a colour pair (LSTM, Transformer) per encoder architecture, a
    hatch per pretraining method and the core markers."""
    def pair(lstm, dec):
        return (Patch(facecolor=lstm, edgecolor='black', linewidth=0.8),
                Patch(facecolor=dec, edgecolor='black', linewidth=0.8))

    styles = S.VISION_TYPE_STYLES
    handles = [pair(styles['AB_LSTM_NOMTURK_U80']['color'], styles['AB_DECODER_NOMTURK_U80']['color'])]
    labels = ['Abstract']
    for label, _, colors in S.VISION_ARCHS.values():
        handles.append(pair(*colors))
        labels.append(label)
    for label, hatch in dict(S.VISION_PRETRAIN.values()).items():
        handles.append(Patch(facecolor='white', edgecolor='black', linewidth=0.8, hatch=hatch))
        labels.append(label)
    for m, label in [('s', 'LSTM'), ('P', 'Transformer')]:
        handles.append(Line2D([], [], marker=m, linestyle='', color='#bdbdbd', markeredgecolor='black',
                              markersize=9))
        labels.append(label)
    fig = plt.figure(figsize=(3.6, 4.2))
    fig.legend(handles, labels, loc='center', frameon=False, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
               handlelength=2.6, handleheight=2.0, fontsize=10)  # tall enough to show each hatch
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)


def size_lines(ax, families, summary, interval):
    """Human band, then one line per family {label: (colour, marker, {x: key})} with error bars
    interval(result) = (mean, low, high)."""
    human_band(ax, *interval(summary['human']))
    for label, (color, marker_, keys) in families.items():
        vals = np.array([interval(summary['models'][k]) for k in keys.values()])
        ax.errorbar(list(keys), vals[:, 0], yerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]],
                    marker=marker_, color=color, linestyle='-', linewidth=2.5, markersize=7, markeredgecolor='black',
                    capsize=4, label=label, zorder=4)


def save_line_legend(ax, path):
    """The families of a size_lines panel, without the human line."""
    save_legend(*zip(*[(h, label) for h, label in zip(*ax.get_legend_handles_labels()) if label != 'Human']), path)


def seq_size():
    keys = S.SEQ_SIZE_KEYS
    data = load('seq_size', keys)
    out = FIGURE_DIR / 'seq_size'
    out.mkdir(parents=True, exist_ok=True)
    for stem, result, mode, interval, ylabel in LINE_PANELS:
        fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
        size_lines(ax, S.SEQ_SIZE_FAMILIES, get_summary_data(data[result], 'overall', keys, mode=mode, agg='overall'),
                   interval)
        ax.set_xscale('log', base=2)
        ax.set_xticks(S.HIDDEN_SIZES)
        ax.set_xticklabels([str(s) for s in S.HIDDEN_SIZES])
        ax.set_ylim(0.0, 1.1)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Hidden Dimension')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.spines[['top', 'right']].set_visible(False)
        fig.savefig(out / f'{stem}_line.png', bbox_inches='tight', dpi=600)
        plt.close(fig)
    save_line_legend(ax, out / 'legend.png')


def encoder_size():
    import timm
    import torch

    def n_params(encoder):  # the headless backbone of the natural model, built on the meta device
        with torch.device('meta'):
            model = timm.create_model(BACKBONES[encoder], pretrained=False, num_classes=0, global_pool='')
        return sum(p.numel() for p in model.parameters())

    keys = [k for _, fam in S.ENCODER_SIZE_FAMILIES.values() for k in fam.values()]
    data = load('encoder_size', keys)
    out = FIGURE_DIR / 'encoder_size'
    out.mkdir(parents=True, exist_ok=True)
    families = {label: (color, 's', {n_params(enc): k for enc, k in fam.items()})
                for label, (color, fam) in S.ENCODER_SIZE_FAMILIES.items()}
    params = {k: n for _, _, fam in families.values() for n, k in fam.items()}
    cons, acc = data['cons_result']['overall'], data['acc_result']['overall']
    for k in keys:
        print(f"[encoder_size] {k:32s} {params[k] / 1e6:7.1f}M  MTurk acc {acc['models'][k]['mean']:.3f}  "
              f"rho(m,h) {cons['models'][k]['mh']['mean']:.3f}", flush=True)

    for stem, result, mode, interval, ylabel in LINE_PANELS:
        fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
        size_lines(ax, families, get_summary_data(data[result], 'overall', keys, mode=mode, agg='overall'), interval)
        ax.set_xscale('log')
        ax.set_xticks([3e7, 1e8, 3e8, 1e9, 3e9])
        ax.set_xticklabels(['30M', '100M', '300M', '1B', '3B'])
        ax.minorticks_off()
        ax.set_ylim(0.0, 1.1)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Vision encoder parameters')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.spines[['top', 'right']].set_visible(False)
        fig.savefig(out / f'{stem}_line.png', bbox_inches='tight', dpi=600)
        plt.close(fig)
    save_line_legend(ax, out / 'legend.png')


# noise level s -> dimensionality-matched LSTM, final checkpoints; models with noise are evaluated
# with 20 seeded noise draws per trial
NOISE_KEYS = {0.0: 'DIM_MATCH_LSTM_NONOISE_NOMTURK_FINAL', 0.25: 'DIM_MATCH_LSTM_NS0.25_NOMTURK',
              0.35: 'DIM_MATCH_LSTM_NOMTURK', 0.4: 'DIM_MATCH_LSTM_NS0.4_NOMTURK',
              0.45: 'DIM_MATCH_LSTM_NS0.45_NOMTURK', 0.5: 'DIM_MATCH_LSTM_NS0.5_NOMTURK'}
DIM_MATCH_COLOR = '#c51b8a'
# seeds stuck on the 0.62 validation-accuracy plateau peak at 0.642 or less, the others reach 0.72 or more
STUCK_VAL = 0.68


def n_stuck(key):
    """Seeds whose validation accuracy never left the 0.62 plateau, from the training runs' validation curves."""
    folder = ALL_DM[key]['folder']
    best = [np.max(np.load(f'{folder}_rep{s}/val_avg_acc_rep{s}.npy', allow_pickle=True)) for s in range(6)]
    assert not any(0.66 < b < 0.70 for b in best), f'{key}: a seed near the stuck threshold, {best}'
    return sum(b < STUCK_VAL for b in best)


def noise_dose():
    keys = list(NOISE_KEYS.values())
    data = load('noise_dose', keys, random=True)
    out = FIGURE_DIR / 'noise_dose'
    out.mkdir(parents=True, exist_ok=True)
    noise = np.array(list(NOISE_KEYS))
    stuck = [n_stuck(k) for k in keys]
    cons, acc = data['cons_result']['overall'], data['acc_result']['overall']

    # error bars as in the bar panels: rho(m,h) mean +/- the split-half std in Fisher z, accuracy
    # mean +/- std over seeds
    def rho(k):
        # the rho(m,h) axis starts at 0: a negative mean and both ends of its bar are drawn at 0
        r, lo, hi = (max(x, 0.0) for x in rho_interval(cons['models'][k]['mh']))
        return r, r - lo, hi - r

    def accuracy(k):
        return acc['models'][k]['mean'], acc['models'][k]['std'], acc['models'][k]['std']

    print('[noise_dose] rho(m,h) before clipping:',
          {n: round(cons['models'][k]['mh']['mean'], 3) for n, k in zip(noise.tolist(), keys)}, flush=True)
    # the random-initialization twins are drawn in the accuracy panel only (their rho(m,h) is below 0)
    for stem, get, ylabel, ylim in [('rho_mh', rho, r'$\rho (m, h)$', (0.0, 1.0)),
                                    ('accuracy', accuracy, 'Accuracy', (0.4, 1.0))]:
        fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
        # human: split-half reliability for rho(m,h), mean accuracy for accuracy
        human_band(ax, *(rho_interval(cons['human']) if stem == 'rho_mh' else std_interval(acc['human'])))
        for suffix, label, kw in [
                ('', 'Dim-matched LSTM', dict(color=DIM_MATCH_COLOR, markerfacecolor=DIM_MATCH_COLOR, linestyle='-')),
                ('_rand', 'Random init', dict(color='#969696', markerfacecolor='white', linestyle=':'))]:
            if suffix == '_rand' and stem == 'rho_mh':
                continue
            vals = np.array([get(k + suffix) for k in keys])
            # clip_on=False keeps the markers sitting on the axis floor whole
            ax.errorbar(noise, vals[:, 0], yerr=[vals[:, 1], vals[:, 2]], marker='s', linewidth=2.5, markersize=7,
                        markeredgecolor='black', capsize=4, label=label, zorder=4 if suffix == '' else 3,
                        clip_on=False, **kw)
        ax.set_xticks(noise)
        ax.set_xticklabels([f'{n:g}' for n in noise])
        ax.set_xlabel('Noise scale')
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, loc='upper right')
        fig.savefig(out / f'{stem}.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
    print('[noise_dose] stuck seeds per level:', dict(zip(noise.tolist(), stuck)), flush=True)


def metric_scatter(x, y, keys, ylabel, ylim, path):
    """One point per model against rho(m,h), with the least-squares line, its bootstrap 95% band and
    the Pearson correlation."""
    df = pd.DataFrame({'X': x, 'Y': y})
    r, p = stats.pearsonr(df['X'], df['Y'])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=12)
    sns.regplot(data=df, x='X', y='Y', scatter=False, truncate=False, seed=0, ax=ax,
                line_kws={'color': 'darkgray', 'lw': 2, 'linestyle': '--', 'zorder': 1})
    for k, xk, yk in zip(keys, x, y):
        core = next(c for c in S.CORES if c in k.split('_'))
        ax.scatter(xk, yk, zorder=2, marker=S.CORE_MARKER[core], facecolors=S.FAMILY_COLOR[S.family(k)],
                   edgecolors='#ffffff', linewidths=0.6, s=100, alpha=0.9)
    ax.text(0.95, 0.05, f'Pearson $r = {r:.3f}$\n$p = {p:.3e}$', transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right', fontsize=14,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.8))
    ax.set_xlabel(r'$\rho(m, h)$')
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# separability panels: x range per feature (linear-classifier accuracy), y range and label per metric
SEPARABILITY_X = {'cat': (0.8, 1.0), 'id': (0, 1.0), 'pos': (0.4, 1.0)}
METRIC_Y = {'acc': (0, 1.0), 'cons': (0, 1.0), 'ec': (-0.2, 0.5)}
METRIC_LABEL = {'acc': 'Task Accuracy', 'cons': r'$\rho(m,h)$', 'ec': r'$\kappa$'}
FEATURE_NAME = {'cat': 'category', 'id': 'identity', 'pos': 'position'}


def separability_scatter(points, feat, metric, path):
    """Each encoder's linear separability of a stimulus feature against a metric of the model trained on
    it, with the least-squares line over the panel's x range and the Pearson correlation."""
    x = np.array([p[1] for p in points])
    y = np.array([p[2] for p in points])
    fig, ax = plt.subplots(figsize=(7, 6))
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(*SEPARABILITY_X[feat], 100)
    ax.plot(x_line, m * x_line + b, color='#333333', linestyle='--', alpha=0.5)
    r, p = stats.pearsonr(x, y)
    ax.text(0.95, 0.05, f'Pearson $r = {r:.3f}$\n$p = {p:.3e}$', transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right', fontsize=14,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.8))
    for fam, xi, yi in points:
        color = S.FAMILY_COLOR[fam]
        # one model per encoder, so the error bar has zero length; its caps are part of the published panels
        ax.errorbar(xi, yi, yerr=0, fmt='none', ecolor=color, elinewidth=1, capsize=2, alpha=0.5)
        ax.scatter(xi, yi, color=color, s=80, edgecolors='white', linewidths=0.5)
    ax.set_xlabel('SGD Accuracy')
    ax.set_ylabel(METRIC_LABEL[metric])
    ax.set_xlim(SEPARABILITY_X[feat])
    ax.set_ylim(METRIC_Y[metric])
    ax.grid(True, linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def metric_alignment():
    keys = S.METRIC_ALIGNMENT_KEYS
    data = load('metric_alignment', keys)
    out = FIGURE_DIR / 'metric_alignment'
    out.mkdir(parents=True, exist_ok=True)

    cons, acc, ec = (data[r]['overall']['models'] for r in ('cons_result', 'acc_result', 'ec_result'))
    rsa = data['rsa_results']
    v = {'corr': [cons[m]['mh']['mean'] for m in keys],
         'kappa': [np.mean(ec[m]['mean']) for m in keys], 'acc': [acc[m]['mean'] for m in keys],
         'task_rsa': [rsa['task_rsa_alignment']['models'][m]['mean'][0] for m in keys],
         'seq_rsa': [rsa['seq_rsa_alignment']['models'][m]['mean'][0] for m in keys]}
    v = {name: np.array(x, dtype=float) for name, x in v.items()}

    # linear separability of each stimulus feature: out-of-fold accuracy of a linear SVM (hinge-loss
    # SGD, 5-fold stratified cross-validation) on each frozen encoder's features of all 5,760 HvM images
    sgd = np.load(EVAL_DIR / 'vision_sgd.npy', allow_pickle=True).item()
    model_of = {BACKBONES[e]: f'{e}_LSTM_NOMTURK' for e in S.SEPARABILITY_ENCODERS}
    assert set(model_of) == set(sgd), set(model_of) ^ set(sgd)
    per_feat = feature_level_results(data)

    with plt.rc_context():
        sns.set_theme(style='whitegrid', context='talk', color_codes=False)
        for y, ylabel, name, ylim in [
                ('kappa', r'$\kappa$', 'rho_kappa', METRIC_Y['ec']),
                ('task_rsa', 'Cross-task RSA', 'rho_task_rsa', (-0.1, 1.0)),
                ('seq_rsa', 'Cross-Stimulus RSA', 'rho_stim_rsa', (-0.1, 1.0)),
                ('acc', 'Accuracy', 'rho_acc', METRIC_Y['acc'])]:
            assert 0 <= v['corr'].min() and v['corr'].max() <= 1.0, 'rho(m,h) outside the x range'
            assert ylim[0] <= v[y].min() and v[y].max() <= ylim[1], f'{name}: values outside {ylim}'
            metric_scatter(v['corr'], v[y], keys, ylabel, ylim, out / f'{name}.png')

        for feat in ['cat', 'id', 'pos']:
            for metric in ['acc', 'cons', 'ec']:
                res = per_feat[FEATURE_NAME[feat]][metric]['overall']['models']
                points = [(S.family(k), sgd[enc][feat]['overall_accuracy'],
                           np.mean([res[k]['mh']['mean'] if metric == 'cons' else res[k]['mean']]))
                          for enc, k in model_of.items()]
                assert all(SEPARABILITY_X[feat][0] <= p[1] <= SEPARABILITY_X[feat][1]
                           and METRIC_Y[metric][0] <= p[2] <= METRIC_Y[metric][1] for p in points), (feat, metric)
                separability_scatter(points, feat, metric, out / f'{feat}_{metric}.png')
                x, y = np.array([p[1] for p in points]), np.array([p[2] for p in points])
                pr, pp = stats.pearsonr(x, y)
                sr, sp = stats.spearmanr(x, y)
                print(f"[metric_alignment] separability {feat} vs {metric}: Pearson r {pr:+.3f} (p {pp:.3g}), "
                      f"Spearman {sr:+.3f} (p {sp:.3g}), n {len(points)}", flush=True)


def _reg_res(donor):
    """Abs->Nat substitutions into the natural LSTM carrying the one-hot abstract LSTM core from its
    final ('final') or 80%-matched ('m80') checkpoint."""
    arm = ARMS['reg_res']
    return {'native': arm['native'], 'theta': arm[f'theta_{donor}donor'],
            'theta_wout': arm[f'theta_wout_{donor}donor'], 'untrained': arm['untrained']}


SUBSTITUTION_SETS = {  # set -> {recipient: {condition: key}}
    'final': {'abs': ARMS['reg_abs_final'], 'nat': _reg_res('final')},
    'acc_matched': {'abs': ARMS['reg_abs_u80'], 'nat': _reg_res('m80')},
    'dm': {'abs': ARMS['dm_abs'], 'nat': ARMS['dm_res']},
}


def transplant(set_name):
    keys_by_recipient = SUBSTITUTION_SETS[set_name]
    models = S.substitution_order(keys_by_recipient)
    styles = S.substitution_styles(keys_by_recipient)
    data = load(f'transplant_{set_name}', models)
    out = FIGURE_DIR / 'transplant' / set_name
    save_bar_panels(overall(data), models, styles, out / 'global', {'rho_mh': 'correlation', 'accuracy': 'accuracy'})

    if set_name == 'final':
        # RSA bars: both panels share a y-range that reaches the lowest 95% CI bound, x-axis line at zero
        rsa = data['rsa_results']
        ci_low = min(rsa[w]['models'][m]['mean'].confidence_interval(confidence_level=0.95).low
                     for w in ('task_rsa_alignment', 'seq_rsa_alignment') for m in models)
        y_lo = min(0.0, np.floor(ci_low / 0.2) * 0.2)
        for stem, which in [('ct_RSA', 'task'), ('cs_RSA', 'seq')]:
            ax = plot_rsa_alignment_summary(data, which, models, styles, ylim=(y_lo, 1.1))
            ax.set_yticks(np.round(np.arange(y_lo, 1.0 + 1e-9, 0.2), 1))
            ax.spines['bottom'].set_position(('data', 0))
            ax.figure.savefig(out / 'global' / f'{stem}.png', dpi=600, bbox_inches='tight')
            plt.close(ax.figure)
    # condition markers in black; colour and brackets give the recipient
    conds = [c for c in S.CONDITIONS if any(c in keys for keys in keys_by_recipient.values())]
    save_legend([marker(S.COND_MARKER[c]) for c in conds], [S.COND_LABEL[c] for c in conds], out / 'legend.png')


SETS = {'seq_type': seq_type, 'ab_match_levels': ab_match_levels, 'vision_type': vision_type, 'seq_size': seq_size,
        'encoder_size': encoder_size, 'noise_dose': noise_dose, 'metric_alignment': metric_alignment,
        **{f'transplant_{s}': (lambda s=s: transplant(s)) for s in SUBSTITUTION_SETS}}

if __name__ == '__main__':
    names = sys.argv[1:] or list(SETS)
    if set(names) - set(SETS):
        sys.exit(f'unknown set(s) {sorted(set(names) - set(SETS))}; sets: {" ".join(SETS)}')
    for name in names:
        SETS[name]()
        print(f'[{name}] done', flush=True)
