"""Paired Natural-vs-Abstract tests of human consistency (Sec. 4.1).

1. The gap in rho(m,h) between the 5 natural ResNet-CLIP models and the 5 one-hot abstract models
   with the same sequence core, paired over cores (primary) and over the 18 task conditions, and the
   gap against the spread across sequence cores and across vision encoders.
2. The same per stimulus feature (category, identity, position).
Abstract models at the 0.80-matched checkpoints (the comparison of Sec. 4.1) and at the final ones.

The test unit is a model: rho(m,h) of each training seed against the mean human response, averaged
over the 6 seeds. This is not the split-half rho(m,h) of the figures (mh_mean), which is printed
next to it in part 2; the two estimators give different values.

usage: PYTHONHASHSEED=0 python scripts/stats/natural_vs_abstract.py
Needs the cached responses in EVAL_DIR (the models are evaluated if missing) and writes the per-seed
values to EVAL_DIR/stats_nomturk_u80_per_seed_rho.csv.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.experiment_pipeline import run_experiment_pipeline  # noqa: E402
from utils.paths import EVAL_DIR  # noqa: E402

# the pipeline orders models through a set of keys, so its split-half values depend on the string hash seed
assert os.environ.get('PYTHONHASHSEED') == '0', 'run with PYTHONHASHSEED=0'

CORES = ['RNN', 'GRU', 'LSTM', 'BERT', 'DECODER']
FEATURES = ['category', 'identity', 'position']
NAT = {c: f'ResNet_CLIP_{c}_NOMTURK' for c in CORES}
AB = {'matched': {c: f'AB_{c}_NOMTURK_U80' for c in CORES},
      'final': {c: f'AB_{c}_NOMTURK_FINAL' for c in CORES}}
# natural models with the two sequence cores trained on every vision encoder
BACKBONES = {core: {bb: f'{bb}_{core}_NOMTURK' for bb in
                    ['ResNet_CLIP', 'ViT_CLIP', 'ResNet_IN1k', 'ViT_IN21k', 'ViT_DINO', 'ViT_DINOv3',
                     'EVA02_IN22k', 'EVA02_CLIP']}
             for core in ['LSTM', 'DECODER']}

ab_keys = [k for arm in AB.values() for k in arm.values()]
nat_keys = list(dict.fromkeys(list(NAT.values()) + [k for d in BACKBONES.values() for k in d.values()]))
data = run_experiment_pipeline(
    experiment_name='natural_vs_abstract', ab_model_keys=ab_keys, nat_model_keys=nat_keys,
    n_splits=20, device='cpu', load_random=False)

conf = data['all_model_conf']
short = {k: len(conf[k]) for k in ab_keys + nat_keys if len(conf[k]) != 6}
assert not short, f'keys without 6 seeds: {short}'
H = np.asarray(data['human_responses'], dtype=float)  # (20 subjects, 1800 trials)
h_mean = H.mean(axis=0)
n_tasks = len(data['new_task_metadata'])
assert H.shape[1] == n_tasks * 100

# {feature: trial indices}, flattened as in run_experiment_pipeline
feat_idx = {f: np.concatenate(list(d.values())).astype(int) for f, d in data['feat_index'].items()}
assert np.array_equal(np.sort(np.concatenate(list(feat_idx.values()))), np.arange(H.shape[1])), \
    'feature indices must partition the trials'


def seed_rhos(key, idx=None):
    """rho(m,h) of each training seed against the mean human response (seeds sorted by id)."""
    idx = np.arange(H.shape[1]) if idx is None else idx
    return np.array([np.corrcoef(np.asarray(v, dtype=float)[idx], h_mean[idx])[0, 1]
                     for _, v in sorted(conf[key].items())])


def paired(name, x, y, labels, n_wilcoxon_min=6):
    """Paired t-test over the units (CI and Cohen's d_z of the difference); Wilcoxon only when n can reach p < 0.05."""
    d = x - y
    t, p = stats.ttest_rel(x, y)
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    print(f'\n--- {name} (n = {len(d)}, paired) ---')
    for l, a, b in zip(labels, x, y):
        print(f'    {str(l):<40s} nat {a:+.4f}  ab {b:+.4f}  diff {a - b:+.4f}')
    line = (f'    Natural {x.mean():+.4f}  Abstract {y.mean():+.4f}  diff {d.mean():+.4f}, '
            f'95% CI [{lo:+.4f}, {hi:+.4f}], t({len(d) - 1}) = {t:.2f}, p = {p:.2e}, '
            f'd_z = {d.mean() / d.std(ddof=1):.2f}, '
            f'Natural > Abstract in {(d > 0).sum()}/{len(d)}')
    if len(d) >= n_wilcoxon_min:
        line += f', Wilcoxon p = {stats.wilcoxon(x, y).pvalue:.2e}'
    print(line)
    return d.mean()


# per-seed table: every key, overall and per feature
rows = []
for key in ab_keys + nat_keys:
    per = {'overall': seed_rhos(key), **{f: seed_rhos(key, feat_idx[f]) for f in FEATURES}}
    for s in range(6):
        rows.append({'key': key, 'seed': s, **{f'rho_{a}': per[a][s] for a in per}})
seed_df = pd.DataFrame(rows)
arch_mean = seed_df.groupby('key').mean(numeric_only=True)
out_csv = EVAL_DIR / 'stats_nomturk_u80_per_seed_rho.csv'
seed_df.to_csv(out_csv, index=False)
print(f'[saved] {out_csv}')

nat_arch = arch_mean.loc[list(NAT.values()), 'rho_overall'].to_numpy()
for ckpt, arm in AB.items():
    ab_arch = arch_mean.loc[list(arm.values()), 'rho_overall'].to_numpy()
    print('\n' + '=' * 100)
    print(f'Natural vs Abstract rho(m,h), abstract {ckpt} checkpoints')
    print('=' * 100)
    gap = paired('By architecture (primary)', nat_arch, ab_arch, CORES)

    task_names = [data['new_task_metadata'][t]['name'][:40] for t in sorted(data['new_task_metadata'])]
    # (n_tasks, n_keys, n_seeds); a seed answering one block with a constant confidence has an undefined rho
    blocks = {name: np.array([[seed_rhos(k, np.arange(i * 100, (i + 1) * 100)) for k in keys.values()]
                              for i in range(n_tasks)]) for name, keys in (('nat', NAT), ('ab', arm))}
    n_const = {name: int(np.isnan(b).sum()) for name, b in blocks.items()}
    if not any(n_const.values()):
        paired('By task block (secondary; mean over architectures and seeds)',
               blocks['nat'].mean(axis=(1, 2)), blocks['ab'].mean(axis=(1, 2)), task_names)
    else:
        print(f'\n[note] constant-confidence seed-blocks: natural {n_const["nat"]}, abstract {n_const["ab"]} '
              f'(of {blocks["ab"].size} each); test run twice')
        paired('By task block (secondary), constant seed-blocks dropped',
               np.nanmean(blocks['nat'], axis=(1, 2)), np.nanmean(blocks['ab'], axis=(1, 2)), task_names)
        paired('By task block (secondary), constant seed-blocks scored rho = 0',
               np.nan_to_num(blocks['nat']).mean(axis=(1, 2)), np.nan_to_num(blocks['ab']).mean(axis=(1, 2)),
               task_names)

    print(f'\n--- Modality gap vs architecture spread (architecture means of per-seed rho) ---')
    print(f'    modality gap (mean paired diff over cores): {gap:+.4f}; '
          f'smallest per-core gap {(nat_arch - ab_arch).min():+.4f}')
    for arm_name, vals in (('natural (ResNet-CLIP)', nat_arch), (f'abstract ({ckpt})', ab_arch)):
        print(f'    sequence cores, {arm_name:<22s}: range {vals.max() - vals.min():.4f} '
              f'({vals.min():+.4f} to {vals.max():+.4f}), SD {vals.std(ddof=1):.4f}')
    for core, bbs in BACKBONES.items():
        vals = arch_mean.loc[list(bbs.values()), 'rho_overall']
        print(f'    vision backbones, natural {core:<8s}: range {vals.max() - vals.min():.4f} '
              f'({vals.idxmin()} {vals.min():+.4f} to {vals.idxmax()} {vals.max():+.4f}), SD {vals.std(ddof=1):.4f}')
    all_nat = arch_mean.loc[nat_keys, 'rho_overall']
    print(f'    lowest natural model {all_nat.idxmin()} {all_nat.min():+.4f} vs highest abstract '
          f'{arch_mean.loc[list(arm.values()), "rho_overall"].idxmax()} {ab_arch.max():+.4f}: '
          f'separation {all_nat.min() - ab_arch.max():+.4f}')

    print('\n' + '=' * 100)
    print(f'Per feature, abstract {ckpt} checkpoints')
    print('=' * 100)
    cons = data['cons_result']['per_feature']
    for f in FEATURES:
        nat_f = arch_mean.loc[list(NAT.values()), f'rho_{f}'].to_numpy()
        ab_f = arch_mean.loc[list(arm.values()), f'rho_{f}'].to_numpy()
        print(f'\n[{f}] {len(feat_idx[f])} trials ({len(feat_idx[f]) / H.shape[1]:.0%}); '
              f'human split-half rho(h,h) {cons[f]["human"]["mean"]:+.3f}')
        print('    split-half rho(m,h) (mh_mean, the figure estimator):')
        for c in CORES:
            n_r, a_r = cons[f]['models'][NAT[c]]['mh'], cons[f]['models'][arm[c]]['mh']
            print(f'      {c:<8s} nat {n_r["mean"]:+.3f} ± {n_r["std"]:.3f}   ab {a_r["mean"]:+.3f} ± {a_r["std"]:.3f}')
        print(f'      mean     nat {np.mean([cons[f]["models"][NAT[c]]["mh"]["mean"] for c in CORES]):+.3f}'
              f'           ab {np.mean([cons[f]["models"][arm[c]]["mh"]["mean"] for c in CORES]):+.3f}')
        paired(f'{f}, by architecture (per-seed rho)', nat_f, ab_f, CORES)
print('\nDONE')
