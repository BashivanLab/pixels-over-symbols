"""The Natural-Abstract rho(m,h) gap after regressing out visual discriminability (appendix section
"Visual discriminability does not explain the behavioral gap" and tab:residualization).

Visual difficulty of a trial (analysis/difficulty.py) is the cosine similarity of its two compared
stimuli in a frozen encoder's cached features (flattened), signed by the ground-truth match and
centred within trial type in each task block. It is regressed out of every human and model response
vector per task block (OLS), and the consistency metrics are recomputed on the residuals. The effect
of difficulty is the residualized gap minus the mean gap over 10 within-block permutations of the
covariate (the shuffled-covariate null). Overall and per stimulus feature.

Covariates: ResNet-CLIP, the features the natural models consume, and DINOv3, an encoder none of the
compared models uses. Gap = mean split-half rho(m,h) (mh_mean) of the 5 natural ResNet-CLIP models
minus that of the 5 one-hot abstract models, at the 0.80-matched and at the final checkpoints.

usage: PYTHONHASHSEED=0 python scripts/stats/discriminability.py
Needs the cached responses in EVAL_DIR, the human responses, the HvM images and metadata, and the
ResNet-CLIP and DINOv3 feature caches in FEATURE_DIR.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.difficulty import (  # noqa: E402
    build_global_pair_info, pair_similarity, difficulty_regressed_consistency, shuffled_covariate_null,
)
from analysis.eval_loop import eval_mturk  # noqa: E402
from analysis.experiment_pipeline import run_experiment_pipeline  # noqa: E402
from analysis.factory import load_emb_cache  # noqa: E402
from analysis.helper import HUMAN_TASK_IDS, get_feature_indices  # noqa: E402
from mturk.stimuli import sample_df  # noqa: E402
from mturk.task_configs import validate_task_configs  # noqa: E402
from utils.paths import HVM_DIR  # noqa: E402
from utils.stim_io import PRETRAINED_STATS  # noqa: E402

# the pipeline orders models through a set of keys, so its split-half values depend on the string hash seed
assert os.environ.get('PYTHONHASHSEED') == '0', 'run with PYTHONHASHSEED=0'

CORES = ['RNN', 'GRU', 'LSTM', 'BERT', 'DECODER']
FEATURES = ['category', 'identity', 'position']
NAT = [f'ResNet_CLIP_{c}_NOMTURK' for c in CORES]
AB = {'matched': [f'AB_{c}_NOMTURK_U80' for c in CORES],
      'final': [f'AB_{c}_NOMTURK_FINAL' for c in CORES]}
# covariate -> a model key whose config names the backbone, i.e. which feature cache to load
COVARIATES = {'ResNet-CLIP': 'ResNet_CLIP_LSTM_NOMTURK', 'DINOv3': 'ViT_DINOv3_LSTM_NOMTURK'}
AGGS = ('per_feature', 'overall')
N_PERM = 10

ab_keys = [k for keys in AB.values() for k in keys]
data = run_experiment_pipeline(
    experiment_name='discriminability', ab_model_keys=ab_keys, nat_model_keys=NAT,
    n_splits=20, device='cpu', load_random=False)
conf = {k: data['all_model_conf'][k] for k in NAT + ab_keys}

# trial -> stimulus pair and ground truth, in the order of the response vectors (human-only evaluation)
task_configs = validate_task_configs()
img_loader, meta_data = sample_df(
    HVM_DIR, 16, 100, 3, df_path=task_configs[1].task_meta_csv, pretrained_stats=PRETRAINED_STATS['clip_resnet'])
_, tpd, fpd, _ = eval_mturk(
    nat_models={}, ab_models={}, vlm_models=[], task_ids=HUMAN_TASK_IDS,
    img_loader=img_loader, meta_data=meta_data, task_configs=task_configs, device=torch.device('cpu'))
flat_fi, _, _ = get_feature_indices(fpd, tpd, task_configs, data['new_task_metadata'], data['new_mapping'],
                                    n_pairs=100)
info = build_global_pair_info(tpd, img_loader, flat_fi)
print(f'[pairs] {len(info["match"])} trials, {info["match"].mean():.3f} match', flush=True)


def gaps(res):
    """{checkpoint: {'overall' | feature: natural minus abstract mean mh_mean}}."""
    def mh(agg, keys):
        ms = res['overall']['models'] if agg == 'overall' else res['per_feature'][agg]['models']
        return np.mean([ms[k]['mh']['mean'] for k in keys])
    return {ckpt: {agg: mh(agg, NAT) - mh(agg, keys) for agg in ('overall',) + tuple(FEATURES)}
            for ckpt, keys in AB.items()}


def run(sim):
    res, _, diag = difficulty_regressed_consistency(
        conf, data['human_responses'], sim, info['match'], data['new_task_metadata'],
        data['model_splits'], data['human_splits'], aggs=AGGS, feature_indices=flat_fi)
    return res, np.nanmean(diag['human_r2_per_block'])


base = gaps(data['cons_result'])
bm = data['cons_result']['overall']['models']
sims = {}
for cov_name, cov_key in COVARIATES.items():
    emb, _ = load_emb_cache(keys=[cov_key], configs=None)[cov_key]
    assert len(emb) == 5760, f'{cov_key}: cache holds {len(emb)} images, expected 5760'
    sims[cov_name] = sim = pair_similarity(info['pairs'], img_loader, emb, pool='flatten')
    del emb

    res, r2 = run(sim)
    resid = gaps(res)
    null = [gaps(run(s)[0]) for s in shuffled_covariate_null(sim, n_permutations=N_PERM, n_pairs=100, random_state=0)]

    print('\n' + '=' * 100)
    print(f'Covariate: {cov_name} ({cov_key} cache), similarity range [{sim.min():.3f}, {sim.max():.3f}], '
          f'R^2 on the mean human response {r2:.4f} (mean over task blocks)')
    print('=' * 100)
    rm = res['overall']['models']
    print(f'{"model":30s} {"base":>9s} {"resid":>9s} {"delta":>8s}   (split-half rho(m,h), overall)')
    for k in NAT + ab_keys:
        print(f'{k:30s} {bm[k]["mh"]["mean"]:+9.4f} {rm[k]["mh"]["mean"]:+9.4f} '
              f'{rm[k]["mh"]["mean"] - bm[k]["mh"]["mean"]:+8.4f}')
    for ckpt in AB:
        print(f'\nNatural minus Abstract ({ckpt}) rho(m,h) gap:')
        for agg in ('overall',) + tuple(FEATURES):
            nv = np.array([n[ckpt][agg] for n in null])
            eff = resid[ckpt][agg] - nv.mean()
            print(f'  {agg:9s} base {base[ckpt][agg]:+.4f}  residualized {resid[ckpt][agg]:+.4f}  '
                  f'null {nv.mean():+.4f} ± {nv.std():.4f}  difficulty effect {eff:+.4f} (z = {eff / nv.std():+.2f})')

a, b = sims.values()
print(f'\ncorr(ResNet-CLIP similarity, DINOv3 similarity) over trials = {np.corrcoef(a, b)[0, 1]:+.3f}')
print('\nDONE')
