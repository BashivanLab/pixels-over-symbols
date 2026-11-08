"""Statistics of the RSA and metric-alignment sentences of Sec. 4.1.

1. Cross-task and cross-stimulus RSA alignment of the 5 natural ResNet-CLIP models against the 5
   one-hot abstract models (0.80-matched checkpoints), paired over the sequence cores: on the
   split-half alignment values of the figures, and with the training seed as the unit (each seed's
   own task x task (18 x 18) or pair x pair (100 x 100) correlation matrix aligned with the human
   split-half matrix by Pearson correlation over the upper triangle, averaged over the 6 seeds).
2. Pearson r across the models of fig:metric_correlation between rho(m,h) (split-half mh_mean) and
   kappa, the cross-stimulus and cross-task RSA alignment, and accuracy.

usage: PYTHONHASHSEED=0 python scripts/stats/rsa.py
Needs the cached responses in EVAL_DIR (the models are evaluated if missing).
"""
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.cross_task import compute_rsa_alignment  # noqa: E402
from analysis.experiment_pipeline import run_experiment_pipeline  # noqa: E402
from utils.const import VLM_MODELS  # noqa: E402

# the pipeline orders models through a set of keys, so its split-half values depend on the string hash seed
assert os.environ.get('PYTHONHASHSEED') == '0', 'run with PYTHONHASHSEED=0'

N_TASKS, N_PAIRS = 18, 100
CORES = ['RNN', 'GRU', 'LSTM', 'BERT', 'DECODER']
# the models of fig:metric_correlation: one-hot abstract models (0.80-matched) with every core and hidden
# size, natural models with every core, vision encoder and hidden size
METRIC_AB = [f'AB_{c}_NOMTURK_U80' for c in CORES] + [
    f'AB_{c}_{h}_NOMTURK_U80' for c in ['LSTM', 'DECODER'] for h in [64, 256, 512, 1024]]
METRIC_NAT = [f'ResNet_CLIP_{c}_NOMTURK' for c in CORES] + [
    'ResNet_IN1k_LSTM_NOMTURK', 'ResNet_IN1k_DECODER_NOMTURK',
    'ViT_IN21k_LSTM_NOMTURK', 'ViT_CLIP_LSTM_NOMTURK', 'ViT_IN21k_DECODER_NOMTURK', 'ViT_CLIP_DECODER_NOMTURK',
    'ViT_DINO_LSTM_NOMTURK', 'ViT_DINOv3_LSTM_NOMTURK', 'ViT_DINO_DECODER_NOMTURK', 'ViT_DINOv3_DECODER_NOMTURK',
    'EVA02_IN22k_LSTM_NOMTURK', 'EVA02_CLIP_LSTM_NOMTURK', 'EVA02_IN22k_DECODER_NOMTURK', 'EVA02_CLIP_DECODER_NOMTURK',
    'EVA02_LARGE_LSTM_NOMTURK', 'EVA02_ENORMOUS_LSTM_NOMTURK',
] + [f'ResNet_CLIP_{c}_{h}_NOMTURK' for c in ['LSTM', 'DECODER'] for h in [64, 256, 512, 1024]]


def load(name, ab, nat, vlm=None):
    """Responses and metrics of the given models from run_experiment_pipeline; prints accuracy and rho(m,h)."""
    np.random.seed(0)
    data = run_experiment_pipeline(name, ab_model_keys=ab, nat_model_keys=nat, vlm_model_keys=vlm,
                                   n_splits=20, device='cpu', load_random=False)
    short = {k: len(data['all_model_conf'][k]) for k in ab + nat if len(data['all_model_conf'][k]) != 6}
    assert not short, f'keys without 6 seeds: {short}'
    acc, cons = data['acc_result']['overall'], data['cons_result']['overall']
    print(f"\n[{name}] {'model':40s} {'acc':>6s} {'rho(m,h)':>9s}", flush=True)
    for k in ab + nat + (vlm or []):
        print(f"[{name}] {k:40s} {acc['models'][k]['mean']:6.3f} {cons['models'][k]['mh']['mean']:9.3f}", flush=True)
    print(f"[{name}] {'human':40s} {acc['human']['mean']:6.3f}", flush=True)
    return data


def seed_rsa(data, key, which):
    """RSA alignment of each seed's own correlation matrix with the human split-half matrix."""
    human = data['rsa_results']['human_cross_task' if which == 'task' else 'human_cross_seq']
    out = []
    for _, v in sorted(data['all_model_conf'][key].items()):
        m = np.asarray(v, dtype=float).reshape(N_TASKS, N_PAIRS)  # task-major, as in the pipeline
        mat = np.corrcoef(m) if which == 'task' else np.corrcoef(m, rowvar=False)
        iu = np.triu_indices(mat.shape[0], k=1)
        ok = np.isfinite(mat[iu])  # a pair (or task) with constant confidence has no correlation
        assert ok.mean() > 0.95, (key, which, ok.mean())
        if not ok.all():
            print(f'    [{key} {which}] {ok.mean():.3f} of the cells defined', flush=True)
        out.append(stats.pearsonr(mat[iu][ok], human[iu][ok])[0] if not ok.all()
                   else compute_rsa_alignment(mat, human)[0])
    return np.array(out)


def paired(label, nat, ab):
    d = nat - ab
    t, p = stats.ttest_rel(nat, ab)
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    print(f'  {label}: Natural {nat.mean():.3f} [{nat.min():.3f}, {nat.max():.3f}]  '
          f'Abstract {ab.mean():.3f} [{ab.min():.3f}, {ab.max():.3f}]  diff {d.mean():+.3f}, '
          f'95% CI [{lo:+.3f}, {hi:+.3f}], t({len(d) - 1}) = {t:.2f}, p = {p:.2e}, '
          f'd_z = {d.mean() / d.std(ddof=1):.2f}, Natural higher {int((d > 0).sum())}/{len(d)}, '
          f'min diff {d.min():+.3f}')


# 1. natural vs abstract RSA alignment, VLMs for reference
ab = [f'AB_{c}_NOMTURK_U80' for c in CORES]
nat = [f'ResNet_CLIP_{c}_NOMTURK' for c in CORES]
data = load('seq_type', ab, nat, vlm=VLM_MODELS)
rsa = data['rsa_results']
for which, name in [('task', 'cross-task'), ('seq', 'cross-stimulus')]:
    print(f'\n=== {name} RSA alignment (Pearson) ===')
    print(f"  {'model':36s} {'split-half (fig)':>16s} {'per-seed mean':>14s} {'seed SD':>8s}  seeds")
    per = {}
    # a VLM run is binary, so up to 12% of its pairs get the same answer in all 18 tasks and have no
    # cross-stimulus correlation; the VLMs are only a reference, so they get the per-seed value for cross-task only
    for k in ab + nat + (VLM_MODELS if which == 'task' else []):
        per[k] = seed_rsa(data, k, which)
        fig = rsa[f'{which}_rsa_alignment']['models'][k]['mean'][0]
        print(f'  {k:36s} {fig:16.3f} {per[k].mean():14.3f} {per[k].std(ddof=1):8.3f}  '
              + ' '.join(f'{x:.3f}' for x in per[k]))
    fig_nat = np.array([rsa[f'{which}_rsa_alignment']['models'][k]['mean'][0] for k in nat])
    fig_ab = np.array([rsa[f'{which}_rsa_alignment']['models'][k]['mean'][0] for k in ab])
    print(f'  paired over the 5 cores ({", ".join(CORES)}):')
    paired('per-seed unit', np.array([per[k].mean() for k in nat]), np.array([per[k].mean() for k in ab]))
    paired('split-half values', fig_nat, fig_ab)

# 2. metric alignment
keys = METRIC_AB + METRIC_NAT
data = load('metric_alignment', METRIC_AB, METRIC_NAT)
cons, acc, ec = (data[r]['overall']['models'] for r in ('cons_result', 'acc_result', 'ec_result'))
rsa = data['rsa_results']
mh = np.array([cons[m]['mh']['mean'] for m in keys])
other = {'kappa': np.array([np.mean(ec[m]['mean']) for m in keys]),
         'cross-stimulus RSA': np.array([rsa['seq_rsa_alignment']['models'][m]['mean'][0] for m in keys]),
         'cross-task RSA': np.array([rsa['task_rsa_alignment']['models'][m]['mean'][0] for m in keys]),
         'accuracy': np.array([acc[m]['mean'] for m in keys])}
is_ab = np.array([k.startswith('AB_') for k in keys])
print(f'\n=== metric alignment: Pearson r with rho(m,h) across models (n = {len(keys)}: '
      f'{is_ab.sum()} abstract, {(~is_ab).sum()} natural) ===')
for name, y in other.items():
    r, p = stats.pearsonr(mh, y)
    rs, ps = stats.spearmanr(mh, y)
    rn, pn = stats.pearsonr(mh[~is_ab], y[~is_ab])
    print(f'  {name:20s} r = {r:.3f} (p = {p:.1e}), Spearman {rs:.3f}; '
          f'natural only r = {rn:.3f} (p = {pn:.1e}, n = {(~is_ab).sum()})')
