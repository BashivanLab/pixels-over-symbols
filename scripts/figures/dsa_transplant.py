"""DSA distance of the substituted models to the recipient's native model against the number of PCA
components.

One panel per recipient and representation: the angular DSA distance between the native models and
each substitution condition, mean +/- SEM over the 12 MTurk tasks and all seed pairs, with the native
seed-to-seed distance as the floor band.

Reads the DSA results of scripts/dsa/run_dsa.py from data/evaluations and writes
figures/dsa/<set>/<rep>_<recipient>.png and figures/dsa/<set>/legend.png, where rep is seq (hidden
state h_t) or pre (input-pathway output z_proj,t), recipient is abs (abstract) or res (natural,
ResNet-CLIP), and set is
  final        one-hot abstract models at final checkpoints, Abs->Nat with the final abstract core
  dm           the dimensionality-matched abstract model
The panels of one representation share their y-range within the dm set, and within final together
with the same panels at the 80%-matched abstract checkpoint (acc_matched), which are computed and
printed but not drawn.

    python scripts/figures/dsa_transplant.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import sem  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.dsa.util import compute_within_cross_3d  # noqa: E402
from analysis.plot import DEFAULT_FIG_SIZE, LINE_COLOR, LINE_STYLE, LINE_WIDTH  # noqa: E402
from analysis.transplant_configs import ARMS  # noqa: E402
from styles import COND_LABEL, COND_MARKER, CONDITIONS, cond_color  # noqa: E402
from utils.paths import EVAL_DIR, FIGURE_DIR  # noqa: E402

DSA_RESULTS = EVAL_DIR / 'dsa_distances.npz'
OUT = FIGURE_DIR / 'dsa'
SUBSTITUTED = [c for c in CONDITIONS if c != 'native']

# (set, recipient) -> (ARMS arm, {condition: ARMS condition}, or None where they are the same)
PANELS = {
    ('final', 'res'): ('reg_res', {'theta': 'theta_finaldonor', 'theta_wout': 'theta_wout_finaldonor',
                                   'untrained': 'untrained'}),
    ('acc_matched', 'res'): ('reg_res', {'theta': 'theta_m80donor', 'theta_wout': 'theta_wout_m80donor',
                                         'untrained': 'untrained'}),
    ('acc_matched', 'abs'): ('reg_abs_u80', None),
    ('final', 'abs'): ('reg_abs_final', None),
    ('dm', 'res'): ('dm_res', None),
    ('dm', 'abs'): ('dm_abs', None),
}
SHARED_YRANGE = [['final', 'acc_matched'], ['dm']]
DRAWN = ['final', 'dm']
RECIPIENT = {'res': 'nat', 'abs': 'abs'}  # colour family in styles.RECIPIENT_COLORS

YLABEL = {'seq': 'DSA angular distance to native',
          'pre': 'DSA angular distance to native (input pathway)'}
# the markers, error bars and reference band of the behavioral bar panels
MARKER = dict(markersize=8, markeredgecolor='black', markeredgewidth=0.8)
ERRBAR = dict(ecolor=LINE_COLOR, capsize=3, elinewidth=1.2)
FLOOR = dict(color='gray', alpha=0.3, zorder=0)
FLOOR_MEAN = dict(color=LINE_COLOR, linestyle=LINE_STYLE, linewidth=LINE_WIDTH, zorder=0)


def mean_sem(v):
    return float(np.nanmean(v)), float(sem(np.asarray(v), axis=None, nan_policy='omit'))


def save(fig, folder, stem):
    (OUT / folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / folder / f'{stem}.png', dpi=600, bbox_inches='tight')


def curves(z, labels, ds, tag):
    """{panel: {'floor': (means, sems), condition: (means, sems)}} over the PCA dimensions ds."""
    stats = {pc: compute_within_cross_3d(z[f'{tag}_pc{pc}'], labels) for pc in ds}
    out = {}
    for panel, (arm, conds) in PANELS.items():
        native = ARMS[arm]['native']
        series = {'floor': [mean_sem(stats[pc][0][native]) for pc in ds]}
        for cond, src in (conds or {c: c for c in SUBSTITUTED}).items():
            if src in ARMS[arm]:
                pair = tuple(sorted((native, ARMS[arm][src])))
                series[cond] = [mean_sem(stats[pc][1][pair]) for pc in ds]
        out[panel] = {k: tuple(np.array(v).T) for k, v in series.items()}
    return out


def plot_panel(ds, tag, panel, series, ylim):
    set_name, recipient = panel
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    fm, fs = series['floor']
    ax.fill_between(ds, fm - fs, fm + fs, **FLOOR)
    ax.plot(ds, fm, **FLOOR_MEAN)
    for cond in SUBSTITUTED:
        if cond not in series:
            continue
        m, s = series[cond]
        ax.errorbar(ds, m, yerr=s, marker=COND_MARKER[cond], color=cond_color(RECIPIENT[recipient], cond),
                    linewidth=1.5, zorder=3, **MARKER, **ERRBAR)
    ax.set_xscale('log')
    ax.set_xticks(ds)
    ax.set_xticklabels([str(d) for d in ds])
    ax.minorticks_off()
    ax.set_ylim(ylim)
    ax.set_xlabel('PCA components', fontsize=12)
    ax.set_ylabel(YLABEL[tag], fontsize=12)
    ax.grid(True, color='0.85', linewidth=0.8)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    save(fig, set_name, f'{tag}_{recipient}')
    plt.close(fig)


def legend(folders, conds=SUBSTITUTED):
    """Condition markers in neutral grey and the floor band, as a file of its own."""
    handles = [Line2D([0], [0], marker=COND_MARKER[c], color='0.55', markerfacecolor='0.75', linewidth=1.5, **MARKER)
               for c in conds]
    names = [COND_LABEL[c] for c in conds]
    handles.append((Patch(color='gray', alpha=0.3),
                    Line2D([0], [0], **{k: v for k, v in FLOOR_MEAN.items() if k != 'zorder'})))
    names.append('native seed-to-seed (floor)')
    fig = plt.figure(figsize=(2.8, 0.4 * len(handles) + 0.4))
    fig.legend(handles, names, loc='center', frameon=False, borderpad=0.6, labelspacing=1.0, handletextpad=0.6,
               handler_map={tuple: HandlerTuple(ndivide=1)})
    for folder in folders:
        save(fig, folder, 'legend')
    plt.close(fig)


def main():
    z = np.load(DSA_RESULTS, allow_pickle=True)
    labels = list(z['labels'])
    ds = [int(x) for x in z['ds']]
    for tag in ('seq', 'pre'):
        data = curves(z, labels, ds, tag)
        for sets in SHARED_YRANGE:
            panels = [p for p in PANELS if p[0] in sets]
            lo = min((m - s).min() for p in panels for m, s in data[p].values())
            hi = max((m + s).max() for p in panels for m, s in data[p].values())
            pad = 0.05 * (hi - lo)
            for p in panels:
                if p[0] in DRAWN:
                    plot_panel(ds, tag, p, data[p], (lo - pad, hi + pad))
        print(f'[{tag}] mean distance to native at PCA 10 (floor first):')
        for p in PANELS:
            print(f"  {'/'.join(p):16s} " + ' '.join(f'{k}={m[ds.index(10)]:.3f}' for k, (m, _) in data[p].items()))
    legend(['dm'])
    legend(['final'], [c for c in SUBSTITUTED if c != 'theta_win'])  # no theta + W_in for the one-hot models
    print('figures written under', OUT)


if __name__ == '__main__':
    main()
