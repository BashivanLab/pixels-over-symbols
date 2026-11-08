"""Keys and configs of the substitution models, and the arms they form.

A substitution freezes the sequence core (theta) of a donor model inside a recipient of the other input
modality. The condition names what else comes from the donor and stays frozen; the rest retrains:

    theta       donor core; input pathway and W_out retrain
    theta_wout  donor core and W_out; the input pathway retrains          (run folders: _frzfcout)
    theta_win   donor core and donor input pathway; W_out retrains        (run folders: _frzinpath)
    untrained   a frozen randomly initialized core; input pathway and W_out retrain   (_randseq)

Dimensionality-matched arms, abstract model DIM_MATCH_LSTM_NOMTURK (noise level 0.35):
    dm_res   Abs->Nat, ResNet-50 CLIP recipient, core of the dimensionality-matched LSTM of the same seed
    dm_vit   Abs->Nat, ViT-B/16 CLIP recipient; no theta_win, since the donor's input pathway takes 49
             tokens of 2048 channels and the ViT gives 197 of 768
    dm_abs   Nat->Abs, dimensionality-matched recipient, core of the natural ResNet-CLIP LSTM
One-hot arms, abstract model the one-hot LSTM:
    reg_res  Abs->Nat, ResNet-50 CLIP recipient, core of the one-hot LSTM at its final checkpoint
             (*_FINALDONOR) or at its checkpoint closest to 0.80 (*_M80DONOR); its untrained core is dm_res's
    reg_abs_final, reg_abs_u80
             Nat->Abs, one-hot recipient, core of the natural ResNet-CLIP LSTM, at the final checkpoints or
             at the checkpoints closest to 0.80, like AB_LSTM_NOMTURK_FINAL and AB_LSTM_NOMTURK_U80
There is no one-hot theta_win: the one-hot input pathway takes 128 inputs, the natural one 171.

No key contains 'rand': the pipelines read a '_rand' suffix as a random-init twin.
"""
from analysis.factory import AB_CONFIG_DICT, BACKBONES, DIM_MATCH_CONFIG_DICT
from utils.paths import CHECKPOINT_DIR

NAT_DIR = CHECKPOINT_DIR / 'abs_to_nat'
ABS_DIR = CHECKPOINT_DIR / 'nat_to_abs'
ABS_U80_DIR = CHECKPOINT_DIR / 'nat_to_abs_until080'
RESNET = BACKBONES['ResNet_CLIP']
VIT = BACKBONES['ViT_CLIP']

_RES_DM = f'{NAT_DIR}/cnntype_{RESNET}_hs128_lstm_lr3e-4_wd1e-4_frozen_dimmatch_nomturk_ls0.2'
_VIT_DM = f'{NAT_DIR}/cnntype_{VIT}_hs128_lstm_lr3e-4_wd1e-4_frozen_dimmatch_nomturk_ls0.2'
_ABS_DM = f'{ABS_DIR}/hs128_lstm_lr3e-4_wd1e-4_dim2048_ns0.35_k0.5_nomturk_frozen'
_RES_REG = f'{NAT_DIR}/cnntype_{RESNET}_hs128_lstm_lr3e-4_wd1e-4_frozen'
_ABS_REG = 'hs128_lstm_lr3e-3_wd1e-4_nomturk_frozen'


def _nat(folder, backbone):
    return {'folder': folder, 'file': 'final_CNNRNN_model', 'backbone': backbone, 'rnn': 'LSTM',
            'hs': 128, 'match_acc': False, 'acc_match': None}


def _dm(folder):
    return {**DIM_MATCH_CONFIG_DICT['DIM_MATCH_LSTM_NOMTURK'], 'folder': folder}


def _ab(name, base_key, root=ABS_DIR):
    return {**AB_CONFIG_DICT[base_key], 'folder': f'{root}/{name}'}


NAT_CONFIGS = {
    'DM_XFER_RES_THETA': _nat(_RES_DM, RESNET),
    'DM_XFER_RES_THETA_WOUT': _nat(f'{_RES_DM}_frzfcout', RESNET),
    'DM_XFER_RES_THETA_WIN': _nat(f'{_RES_DM}_frzinpath', RESNET),
    'DM_XFER_RES_UNTRAINED': _nat(f'{_RES_DM}_randseq', RESNET),
    'DM_XFER_VIT_THETA': _nat(_VIT_DM, VIT),
    'DM_XFER_VIT_THETA_WOUT': _nat(f'{_VIT_DM}_frzfcout', VIT),
    'DM_XFER_VIT_UNTRAINED': _nat(f'{_VIT_DM}_randseq', VIT),
    'REG_XFER_RES_THETA_FINALDONOR': _nat(f'{_RES_REG}_abfinal_nomturk_ls0.2', RESNET),
    'REG_XFER_RES_THETA_WOUT_FINALDONOR': _nat(f'{_RES_REG}_abfinal_nomturk_ls0.2_frzfcout', RESNET),
    'REG_XFER_RES_THETA_M80DONOR': _nat(f'{_RES_REG}_abm80_nomturk_ls0.2', RESNET),
    'REG_XFER_RES_THETA_WOUT_M80DONOR': _nat(f'{_RES_REG}_abm80_nomturk_ls0.2_frzfcout', RESNET),
}

DM_CONFIGS = {
    'DM_XFER_ABS_THETA': _dm(f'{_ABS_DM}_nath8_ls0.2'),
    'DM_XFER_ABS_THETA_WOUT': _dm(f'{_ABS_DM}_nath8_ls0.2_frzfcout'),
    'DM_XFER_ABS_THETA_WIN': _dm(f'{_ABS_DM}_nath8_ls0.2_frzinpath'),
    'DM_XFER_ABS_UNTRAINED': _dm(f'{_ABS_DM}_nat_ls0.2_randseq'),
}

# checkpoints selected as for the one-hot LSTM of the same key suffix
AB_CONFIGS = {}
for _suffix in ['_FINAL', '_U80']:
    _root = ABS_U80_DIR if _suffix == '_U80' else ABS_DIR
    _base = f'AB_LSTM_NOMTURK{_suffix}'
    AB_CONFIGS[f'REG_XFER_ABS_THETA{_suffix}'] = _ab(f'{_ABS_REG}_nath8', _base, _root)
    AB_CONFIGS[f'REG_XFER_ABS_THETA_WOUT{_suffix}'] = _ab(f'{_ABS_REG}_nath8_frzfcout', _base, _root)
    AB_CONFIGS[f'REG_XFER_ABS_UNTRAINED{_suffix}'] = _ab(f'{_ABS_REG}_randseq', _base, _root)

ARMS = {
    'dm_res': {'native': 'ResNet_CLIP_LSTM_NOMTURK', 'theta': 'DM_XFER_RES_THETA',
               'theta_wout': 'DM_XFER_RES_THETA_WOUT', 'theta_win': 'DM_XFER_RES_THETA_WIN',
               'untrained': 'DM_XFER_RES_UNTRAINED'},
    'dm_vit': {'native': 'ViT_CLIP_LSTM_NOMTURK', 'theta': 'DM_XFER_VIT_THETA',
               'theta_wout': 'DM_XFER_VIT_THETA_WOUT', 'untrained': 'DM_XFER_VIT_UNTRAINED'},
    'dm_abs': {'native': 'DIM_MATCH_LSTM_NOMTURK', 'theta': 'DM_XFER_ABS_THETA',
               'theta_wout': 'DM_XFER_ABS_THETA_WOUT', 'theta_win': 'DM_XFER_ABS_THETA_WIN',
               'untrained': 'DM_XFER_ABS_UNTRAINED'},
    'reg_res': {'native': 'ResNet_CLIP_LSTM_NOMTURK',
                'theta_finaldonor': 'REG_XFER_RES_THETA_FINALDONOR',
                'theta_wout_finaldonor': 'REG_XFER_RES_THETA_WOUT_FINALDONOR',
                'theta_m80donor': 'REG_XFER_RES_THETA_M80DONOR',
                'theta_wout_m80donor': 'REG_XFER_RES_THETA_WOUT_M80DONOR',
                'untrained': 'DM_XFER_RES_UNTRAINED'},
    'reg_abs_final': {'native': 'AB_LSTM_NOMTURK_FINAL', 'theta': 'REG_XFER_ABS_THETA_FINAL',
                      'theta_wout': 'REG_XFER_ABS_THETA_WOUT_FINAL',
                      'untrained': 'REG_XFER_ABS_UNTRAINED_FINAL'},
    'reg_abs_u80': {'native': 'AB_LSTM_NOMTURK_U80', 'theta': 'REG_XFER_ABS_THETA_U80',
                    'theta_wout': 'REG_XFER_ABS_THETA_WOUT_U80', 'untrained': 'REG_XFER_ABS_UNTRAINED_U80'},
}
