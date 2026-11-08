"""The systems of the paper's DSA and their cached hidden states.

A system is a model key of ``analysis.factory`` or ``analysis.transplant_configs``; each of
its 6 seeds is compared separately. The list order fixes the order of the pairs in the batched
alignment and with it each pair's random initialization (see ``analysis.dsa.simdist.optimize_C``),
so adding, removing or reordering systems changes every distance a little.
"""

import numpy as np

from analysis.experiment_pipeline import embedding_pipeline
from analysis.factory import AB_CONFIG_DICT, DIM_MATCH_CONFIG_DICT, NAT_CONFIG_DICT
from analysis.transplant_configs import AB_CONFIGS, DM_CONFIGS, NAT_CONFIGS
from utils.paths import EVAL_DIR

AB = {**AB_CONFIG_DICT, **AB_CONFIGS}
DM = {**DIM_MATCH_CONFIG_DICT, **DM_CONFIGS}
NAT = {**NAT_CONFIG_DICT, **NAT_CONFIGS}

SYSTEMS = [
    # Abs->Nat into ResNet-CLIP with the dimensionality-matched core (noise level 0.35), and the native
    'ResNet_CLIP_LSTM_NOMTURK', 'DM_XFER_RES_THETA', 'DM_XFER_RES_THETA_WOUT', 'DM_XFER_RES_THETA_WIN',
    'DM_XFER_RES_UNTRAINED',
    # the same into ViT-CLIP; no figure shows them, but leaving them out would change all other distances
    'ViT_CLIP_LSTM_NOMTURK', 'DM_XFER_VIT_THETA', 'DM_XFER_VIT_THETA_WOUT', 'DM_XFER_VIT_UNTRAINED',
    # Nat->Abs into the dimensionality-matched model, and the native
    'DIM_MATCH_LSTM_NOMTURK', 'DM_XFER_ABS_THETA', 'DM_XFER_ABS_THETA_WOUT', 'DM_XFER_ABS_THETA_WIN',
    'DM_XFER_ABS_UNTRAINED',
    # Abs->Nat into ResNet-CLIP with the one-hot LSTM core, from the final and the 0.80-matched donor
    'REG_XFER_RES_THETA_FINALDONOR', 'REG_XFER_RES_THETA_WOUT_FINALDONOR',
    'REG_XFER_RES_THETA_M80DONOR', 'REG_XFER_RES_THETA_WOUT_M80DONOR',
    # Nat->Abs into the one-hot LSTM, and the native, at the 0.80-matched and the final checkpoints
    'AB_LSTM_NOMTURK_U80', 'REG_XFER_ABS_THETA_U80', 'REG_XFER_ABS_THETA_WOUT_U80', 'REG_XFER_ABS_UNTRAINED_U80',
    'AB_LSTM_NOMTURK_FINAL', 'REG_XFER_ABS_THETA_FINAL', 'REG_XFER_ABS_THETA_WOUT_FINAL',
    'REG_XFER_ABS_UNTRAINED_FINAL',
]


def feature_path(key, tag):
    """Cache file of ``key``'s hidden states, ``tag`` 'seq' (h_t) or 'pre' (z_proj), named as by the evaluation."""
    cfg = {**NAT, **AB, **DM}[key]
    stem = f"{key}_match{cfg['acc_match']}" if cfg['match_acc'] else key
    return EVAL_DIR / f'{stem}_{tag}_feature.npy'


def load_features(keys, tag):
    """``{key -> {seed -> {task_id -> (T, N, H) array}}}`` of every key's cached ``tag`` states.

    Missing caches are first computed from the checkpoints.
    """
    missing = [k for k in keys if not (feature_path(k, 'seq').is_file() and feature_path(k, 'pre').is_file())]
    if missing:
        embedding_pipeline(
            'dsa',
            ab_model_keys=[k for k in missing if k in AB],
            nat_model_keys=[k for k in missing if k not in AB and k not in DM],
            dim_match_model_keys=[k for k in missing if k in DM],
            custom_nat_configs=NAT_CONFIGS, custom_ab_configs=AB_CONFIGS, custom_dim_match_configs=DM_CONFIGS,
        )
    features = {k: np.load(feature_path(k, tag), allow_pickle=True).item() for k in keys}
    short = {k: len(v) for k, v in features.items() if len(v) != 6}
    assert not short, f'systems without 6 seeds: {short}'
    return features
