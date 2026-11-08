"""Tests the DSA comparisons of the paper in every configuration of scripts/dsa/sensitivity.py.

Prints tab:dsa_sensitivity and the numbers of the appendix paragraph on the sensitivity to the DSA
settings: for each comparison the configurations in which it holds, the distances behind the
percentages, the default-setting p-values the text quotes, and how much the optimizer settings and
seeds move the task-averaged distances.

A comparison is a one-sided paired Wilcoxon signed-rank test over the 12 tasks (p < 0.05). Per task,
the distance between two models is the mean over all pairs of their seeds, and a model's seed floor
is the mean over the pairs of its own seeds. "Within the floor" holds when the distance is not
significantly above the floor.

usage: python scripts/dsa/sensitivity_report.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.dsa.util import compute_within_cross_3d  # noqa: E402
from utils.paths import EVAL_DIR  # noqa: E402

# the 11 configurations that change a DSA setting, by axis (the default, 'base', counts under mu),
# and the 8 that change only the optimizer settings or the seed
AXES = {'mu': ['pc2', 'pc5', 'base', 'pc20', 'pc50'], 'delays': ['delays1', 'delays3'],
        'rank': ['rank2', 'rank4'], 'distance': ['euclidean', 'wasserstein']}
SETTINGS = [c for cfgs in AXES.values() for c in cfgs]
RERUNS = ['it500', 'it1000', 'it1000_lr1e-3', 'it1000_lr1e-2', 'seed0', 'seed7', 'seed42', 'seed2025']

# native models: natural ResNet-CLIP, one-hot abstract (final and 0.80-matched), dimensionality-matched
NAT, ABS, ABS80 = 'ResNet_CLIP_LSTM_NOMTURK', 'AB_LSTM_NOMTURK_FINAL', 'AB_LSTM_NOMTURK_U80'
DIM = 'DIM_MATCH_LSTM_NOMTURK'
# Abs->Nat substitutions: one-hot core (final donor), dimensionality-matched core, untrained core
A2N, A2N_WOUT = 'REG_XFER_RES_THETA_FINALDONOR', 'REG_XFER_RES_THETA_WOUT_FINALDONOR'
DIM_A2N, DIM_A2N_WOUT, DIM_A2N_WIN = 'DM_XFER_RES_THETA', 'DM_XFER_RES_THETA_WOUT', 'DM_XFER_RES_THETA_WIN'
A2N_UNTRAINED = 'DM_XFER_RES_UNTRAINED'
# Nat->Abs substitutions: one-hot recipient (final; untrained core also 0.80-matched), dimensionality-matched recipient
N2A, N2A_WOUT = 'REG_XFER_ABS_THETA_FINAL', 'REG_XFER_ABS_THETA_WOUT_FINAL'
N2A_UNTRAINED = 'REG_XFER_ABS_UNTRAINED_FINAL'
N2A80_UNTRAINED = 'REG_XFER_ABS_UNTRAINED_U80'
DIM_N2A, DIM_N2A_WOUT, DIM_N2A_WIN, DIM_N2A_UNTRAINED = (
    'DM_XFER_ABS_THETA', 'DM_XFER_ABS_THETA_WOUT', 'DM_XFER_ABS_THETA_WIN', 'DM_XFER_ABS_UNTRAINED')


def floor(f, model, native):
    """``model`` within ``native``'s seed floor on feature ``f``."""
    return 'floor', f, model, native


def above(f, model, native):
    """``model`` significantly above ``native``'s seed floor."""
    return 'above', f, model, native


def farther(f, a, b, native):
    """``a`` significantly farther from ``native`` than ``b``."""
    return 'farther', f, a, b, native


# (where in the paper, statement, test, holds from 5 PCA components on only)
CLAIMS = [
    ('Sec. 4.3', 'theta Abs->Nat within the Natural floor', floor('seq', A2N, NAT), False),
    ('Sec. 4.3', 'untrained theta farther from the natives than theta Abs->Nat',
     farther('seq', A2N_UNTRAINED, A2N, NAT), False),
    ('Sec. 4.3', 'untrained theta above the Natural floor', above('seq', A2N_UNTRAINED, NAT), False),
    ('Sec. 4.3', 'z_proj of theta Abs->Nat above the Natural floor', above('pre', A2N, NAT), False),
    ('Sec. 4.3', 'z_proj of (theta+W_out) Abs->Nat within the Natural floor', floor('pre', A2N_WOUT, NAT), False),
    ('Sec. 4.3', '(theta+W_out) Abs->Nat above the Natural floor', above('seq', A2N_WOUT, NAT), False),
    ('Sec. 4.3', '(theta+W_out) Abs->Nat farther from the natives than theta',
     farther('seq', A2N_WOUT, A2N, NAT), False),
    ('Sec. 4.3', 'dim.-matched theta Abs->Nat within the Natural floor', floor('seq', DIM_A2N, NAT), False),
    ('Sec. 4.3', 'dim.-matched (theta+W_out) Abs->Nat within the Natural floor',
     floor('seq', DIM_A2N_WOUT, NAT), False),
    ('Sec. 4.3', 'dim.-matched theta+W_in above the Natural floor', above('seq', DIM_A2N_WIN, NAT), False),
    ('Sec. 4.3', 'dim.-matched theta+W_in above the dim.-matched floor', above('seq', DIM_N2A_WIN, DIM), False),
    ('Sec. 4.3', 'dim.-matched theta+W_in farther than untrained theta, natural input',
     farther('seq', DIM_A2N_WIN, A2N_UNTRAINED, NAT), False),
    ('Sec. 4.3', 'dim.-matched theta+W_in farther than untrained theta, abstract input',
     farther('seq', DIM_N2A_WIN, DIM_N2A_UNTRAINED, DIM), False),
    ('Sec. 4.3', 'theta Nat->Abs above the Abstract floor', above('seq', N2A, ABS), False),
    ('Sec. 4.3', 'dim.-matched theta Nat->Abs above the dim.-matched floor', above('seq', DIM_N2A, DIM), False),
    ('Sec. 4.3', 'z_proj of theta Nat->Abs within the Abstract floor', floor('pre', N2A, ABS), False),
    ('Sec. 4.3', 'z_proj of dim.-matched theta Nat->Abs within the dim.-matched floor',
     floor('pre', DIM_N2A, DIM), False),
    ('Sec. 4.3', 'untrained theta farther from the Abstract natives than theta Nat->Abs',
     farther('seq', N2A_UNTRAINED, N2A, ABS), False),
    ('Sec. 4.3', 'untrained theta farther from the dim.-matched natives than theta Nat->Abs',
     farther('seq', DIM_N2A_UNTRAINED, DIM_N2A, DIM), False),
    ('Fig. 4a', 'theta+W_out above the Abstract floor', above('seq', N2A_WOUT, ABS), False),
    ('Fig. 4a', 'untrained theta farther than theta+W_out', farther('seq', N2A_UNTRAINED, N2A_WOUT, ABS), False),
    ('Fig. 4b', 'theta within the Natural floor', floor('seq', A2N, NAT), True),
    ('Fig. 4b', 'theta+W_out farther than untrained theta', farther('seq', A2N_WOUT, A2N_UNTRAINED, NAT), True),
    ('z_proj fig. (a)', 'theta+W_out above the Abstract floor', above('pre', N2A_WOUT, ABS), False),
    ('z_proj fig. (a)', 'untrained theta above the Abstract floor', above('pre', N2A_UNTRAINED, ABS), False),
    ('z_proj fig. (b)', 'theta above the Natural floor', above('pre', A2N, NAT), False),
    ('z_proj fig. (b)', 'theta farther than theta+W_out', farther('pre', A2N, A2N_WOUT, NAT), False),
    ('z_proj fig. (b)', 'untrained theta farther than theta', farther('pre', A2N_UNTRAINED, A2N, NAT), False),
    ('dim.-matched fig. (a)', 'theta+W_out above the dim.-matched floor', above('pre', DIM_N2A_WOUT, DIM), False),
    ('dim.-matched fig. (a)', 'untrained theta above the dim.-matched floor',
     above('pre', DIM_N2A_UNTRAINED, DIM), False),
    ('dim.-matched fig. (a)', 'theta+W_in farther than untrained theta',
     farther('pre', DIM_N2A_WIN, DIM_N2A_UNTRAINED, DIM), False),
    ('dim.-matched fig. (b)', 'theta within the Natural floor', floor('pre', DIM_A2N, NAT), True),
    ('dim.-matched fig. (b)', 'theta+W_out within the Natural floor', floor('pre', DIM_A2N_WOUT, NAT), True),
    ('dim.-matched fig. (b)', 'theta+W_in above the Natural floor', above('pre', DIM_A2N_WIN, NAT), False),
    ('dim.-matched fig. (b)', 'untrained theta farther than theta+W_in',
     farther('pre', A2N_UNTRAINED, DIM_A2N_WIN, NAT), False),
    ('dim.-matched fig. (c)', 'theta+W_out above the dim.-matched floor', above('seq', DIM_N2A_WOUT, DIM), False),
    ('dim.-matched fig. (c)', 'untrained theta farther than theta+W_out',
     farther('seq', DIM_N2A_UNTRAINED, DIM_N2A_WOUT, DIM), False),
    ('dim.-matched fig. (d)', 'theta within the Natural floor', floor('seq', DIM_A2N, NAT), True),
    ('dim.-matched fig. (d)', 'theta+W_out within the Natural floor', floor('seq', DIM_A2N_WOUT, NAT), True),
    ('dim.-matched fig. (d)', 'untrained theta above the Natural floor', above('seq', A2N_UNTRAINED, NAT), False),
    ('App. matched vs. final', 'untrained theta within the floor of the 0.80-matched Abstract natives',
     floor('seq', N2A80_UNTRAINED, ABS80), False),
]

TH, WO = r'\theta', r'\mathbf{W}_{\mathrm{out}}'
AN, NA = r'_{\mathrm{Abs}\to\mathrm{Nat}}', r'_{\mathrm{Nat}\to\mathrm{Abs}}'
H, Z = r'$\mathbf{h}_t$', r'$\mathbf{z}_{\mathrm{proj}_t}$'
TABLE_ROWS = [
    (r'\multicolumn{7}{l}{\emph{Natural input}} \\', None, None),
    (f'${TH}{AN}$ within floor', H, [floor('seq', A2N, NAT)]),
    (f'Untrained ${TH}$ farther than ${TH}{AN}$', H, [farther('seq', A2N_UNTRAINED, A2N, NAT)]),
    (f'${TH}{AN}$ above floor', Z, [above('pre', A2N, NAT)]),
    (f'$({TH}+{WO}){AN}$ within floor', Z, [floor('pre', A2N_WOUT, NAT)]),
    (f'Dim.-matched ${TH}$, ${TH}+{WO}$ within floor', H,
     [floor('seq', DIM_A2N, NAT), floor('seq', DIM_A2N_WOUT, NAT)]),
    (r'\midrule \multicolumn{7}{l}{\emph{Abstract input}} \\', None, None),
    (f'${TH}{NA}$ within floor', Z, [floor('pre', N2A, ABS)]),
]
TABLE_HEAD = r'''\begin{table}[ht]
\centering
\caption{\textbf{DSA sensitivity, number of configurations in which each comparison holds.} Each column varies one setting around the default setting of the main text ($\mu=10$, 2 delays, rank 3, angular distance), which is counted once, in the $\mu$ column. The floor is the mean distance between seeds of the native model of the input modality; ``within'' means not significantly above it (one-sided paired Wilcoxon signed-rank test over the 12 tasks, $p<0.05$). The 8 optimizer and seed configurations give the default outcome in every row and are not shown.}
\label{tab:dsa_sensitivity}
\small
\begin{tabular}{lcccccc}
\toprule
Comparison & Feature & $\mu$ & Delays & Rank & Distance & Total \\
 & & 2--50 & 1, 3 & 2, 4 & Eucl./Wass. & \\
\midrule'''
TABLE_TAIL = r'''\bottomrule
\end{tabular}
\end{table}'''


def task_means(cfg):
    """{feature: ({model: per-task floor}, {(model, model): per-task distance})} of one configuration."""
    z = np.load(EVAL_DIR / 'dsa_sensitivity' / f'{cfg}.npz', allow_pickle=True)
    out = {}
    for f in ('seq', 'pre'):
        within, between = compute_within_cross_3d(z[f].astype(np.float64), list(z['labels']))
        out[f] = ({k: np.nanmean(v, axis=1) for k, v in within.items()},
                  {k: np.nanmean(v, axis=1) for k, v in between.items()})
    return out


STATS = {cfg: task_means(cfg) for cfg in SETTINGS + RERUNS}


def floor_of(cfg, f, model):
    return STATS[cfg][f][0][model]


def cross(cfg, f, a, b):
    return STATS[cfg][f][1][tuple(sorted((a, b)))]


def greater(x, y):
    """One-sided paired Wilcoxon signed-rank test over tasks: p-value of x > y."""
    return wilcoxon(x, y, alternative='greater')[1]


def holds(test, cfg):
    kind, f = test[:2]
    if kind == 'floor':
        return not greater(cross(cfg, f, test[2], test[3]), floor_of(cfg, f, test[3])) < 0.05
    if kind == 'above':
        return greater(cross(cfg, f, test[2], test[3]), floor_of(cfg, f, test[3])) < 0.05
    if kind == 'farther':
        return greater(cross(cfg, f, test[2], test[4]), cross(cfg, f, test[3], test[4])) < 0.05
    raise ValueError(kind)


def print_table():
    print(TABLE_HEAD)
    for label, feature, tests in TABLE_ROWS:
        if tests is None:
            print(label)
            continue
        ok = {c: all(holds(t, c) for t in tests) for c in SETTINGS + RERUNS}
        assert all(ok[c] == ok['base'] for c in RERUNS), f'an optimizer or seed rerun changes: {label}'
        cells = [[ok[c] for c in cfgs] for cfgs in AXES.values()]
        total = sum(map(sum, cells))
        print(f'{label} & {feature} & ' + ' & '.join(f'{sum(x)}/{len(x)}' for x in cells)
              + f' & {total}/{len(SETTINGS)} \\\\')
    print(TABLE_TAIL)


def print_claims():
    print('\nIn how many configurations each comparison holds (per axis, the 11 settings, the 8 optimizer and '
          'seed reruns), and where it fails. Comparisons marked * are stated from 5 PCA components on and '
          'leave out mu = 2.')
    print('| Where | Comparison | Feature | mu | delays | rank | distance | settings | reruns | fails in |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for where, text, test, from5 in CLAIMS:
        keep = [c for c in SETTINGS + RERUNS if not (from5 and c == 'pc2')]
        ok = {c: holds(test, c) for c in keep}
        axes = [[ok[c] for c in cfgs if c in ok] for cfgs in AXES.values()]
        settings = [ok[c] for c in SETTINGS if c in ok]
        reruns = [ok[c] for c in RERUNS]
        print(f"| {where} | {text}{' *' if from5 else ''} | {'h_t' if test[1] == 'seq' else 'z_proj'} | "
              + ' | '.join(f'{sum(a)}/{len(a)}' for a in axes)
              + f' | {sum(settings)}/{len(settings)} | {sum(reruns)}/{len(reruns)} | '
              + (', '.join(c for c in keep if not ok[c]) or '-') + ' |')


def print_distances():
    print('\nMean h_t distances over tasks and seed pairs, to the Natural natives (natural input) and to the '
          'final one-hot Abstract natives (abstract input). excess = (theta Abs->Nat - floor) / (untrained - floor): '
          '0 at the Natural floor, 1 as far as the untrained theta.')
    print(f'{"":15s} {"natural input":>40s}   {"abstract input":>26s}')
    print(f'{"config":15s} {"floor":>7s} {"theta":>7s} {"excess":>7s} {"untrained":>9s}   '
          f'{"floor":>7s} {"theta":>7s} {"untrained":>9s}')
    for cfg in SETTINGS + RERUNS:
        fn, a2n, un = floor_of(cfg, 'seq', NAT), cross(cfg, 'seq', A2N, NAT), cross(cfg, 'seq', A2N_UNTRAINED, NAT)
        fa, n2a, n2a_un = floor_of(cfg, 'seq', ABS), cross(cfg, 'seq', N2A, ABS), cross(cfg, 'seq', N2A_UNTRAINED, ABS)
        excess = (a2n.mean() - fn.mean()) / (un.mean() - fn.mean())
        print(f'{cfg:15s} {fn.mean():7.3f} {a2n.mean():7.3f} {excess:7.2f} {un.mean():9.3f}   '
              f'{fa.mean():7.3f} {n2a.mean():7.3f} {n2a_un.mean():9.3f}')


def print_default_detail():
    print('\nAt the default setting (h_t, 10 components), per task:')
    # the 12 MTurk tasks in the order of the distance tensors (task ids 1, 2, 3, 19, 23, 27, 28, 32, 36, 37, 40, 41)
    tasks = ['1back_pos', '1back_id', '1back_cat', 'ABAB_pos', 'ABAB_id', 'ABAB_cat', 'ABBA_pos', 'ABBA_id',
             'ABBA_cat', 'ctxdms_pci', 'ctxdms_cip', 'dms_pos']
    for text, a, b, native in [
            ('dim.-matched theta+W_in vs untrained theta, natural input', DIM_A2N_WIN, A2N_UNTRAINED, NAT),
            ('dim.-matched theta+W_in vs untrained theta, abstract input', DIM_N2A_WIN, DIM_N2A_UNTRAINED, DIM),
            ('untrained theta vs theta Nat->Abs', N2A_UNTRAINED, N2A, ABS)]:
        x, y = cross('base', 'seq', a, native), cross('base', 'seq', b, native)
        print(f'  {text}: {x.mean():.3f} vs {y.mean():.3f}, larger in {(x > y).sum()}/12 tasks, '
              f'one-sided p = {greater(x, y):.3f}; not larger in: '
              + ', '.join(t for t, d in zip(tasks, x - y) if d <= 0))


def print_rerun_change():
    """Largest change of a task-averaged distance (any floor or cross distance) from the default."""
    def means(cfg, f):
        within, between = STATS[cfg][f]
        return {**{('floor', k): v.mean() for k, v in within.items()}, **{k: v.mean() for k, v in between.items()}}

    print('\nLargest change of a task-averaged distance in the optimizer and seed reruns:')
    for f in ('seq', 'pre'):
        base = means('base', f)
        worst = max(abs(means(c, f)[k] - v) for c in RERUNS for k, v in base.items())
        print(f"  {'h_t' if f == 'seq' else 'z_proj'}: {worst:.4f}")


if __name__ == '__main__':
    print_table()
    print_claims()
    print_distances()
    print_default_detail()
    print_rerun_change()
