"""Model configs of the paper and the rebuilding of the trained models for evaluation.

A config key names one model trained with six seeds (rep 0-5, initialization seed 2025 + rep). The keys
end in _NOMTURK because every model was trained on the split that holds out the 100 MTurk stimuli. A
config gives:
    folder, suffix   run folder of rep r: f'{folder}_rep{r}{suffix}' (see run_dir)
    file             name prefix of the final checkpoint, f'{file}_rep{r}.pth'
    rnn, hs          sequence core and hidden size
    match_acc        False: final checkpoint; True: the checkpoint whose validation accuracy is closest
                     to acc_match, among the epoch-0 checkpoints, or among all with any_epoch
The configs are AB_CONFIG_DICT (one-hot abstract models, CustomRNN), NAT_CONFIG_DICT (natural models,
CNNRNN on cached frozen-encoder features; backbone is the timm encoder) and DIM_MATCH_CONFIG_DICT
(dimensionality-matched abstract models, DimMatchCNNRNN). The Abs->Nat and Nat->Abs substitution models
are in analysis/transplant_configs.py.
"""

import logging
from pathlib import Path

import torch

from utils.paths import CHECKPOINT_DIR, FEATURE_DIR
from models.model import CustomRNN, CNNRNN
from models.dim_match import DimMatchCNNRNN, build_projection
from mturk.trials import cached_feat_to_tokens
from utils.const import INPUT_SIZE, TASK_NAME_TASK_INDEX
from utils.helper import seed_everything, load_repair_checkpoint, find_emb_cache

logger = logging.getLogger(__name__)

# Every model in the paper was trained with 8 attention heads (attention pooling and Transformer cores).
# The head count is not stored in a checkpoint: the attention weights have the same shapes for any head
# count, so a wrong value loads without error and only changes the outputs.
NUM_HEADS = 8
DEFAULT_MATCH_ACC = 0.75

BACKBONES = {
    'ResNet_IN1k': 'resnet50.a1_in1k',
    'ResNet_CLIP': 'resnet50_clip.openai',
    'ViT_IN21k': 'vit_base_patch16_224.orig_in21k',
    'ViT_CLIP': 'vit_base_patch16_clip_224.laion2b_ft_in1k',
    'ViT_DINO': 'vit_base_patch16_224.dino',
    'ViT_DINOv3': 'vit_base_patch16_dinov3.lvd1689m',
    'EVA02_IN22k': 'eva02_base_patch14_224.mim_in22k',
    'EVA02_CLIP': 'eva02_base_patch16_clip_224.merged2b',
    'EVA02_LARGE': 'eva02_large_patch14_clip_224.merged2b',
    'EVA02_ENORMOUS': 'eva02_enormous_patch14_clip_224.laion2b_plus',
    **{f'ConvNeXt_{size.upper()}': f'convnext_{size}.fb_in22k' for size in ['tiny', 'small', 'base', 'large', 'xlarge']},
}

CORES = ['RNN', 'GRU', 'LSTM', 'DECODER', 'BERT']
# hidden size of each sequence core in the core comparison ('DECODER' is the causal Transformer)
HIDDEN_DICT = {'RNN': 256, 'GRU': 128, 'LSTM': 128, 'DECODER': 128, 'BERT': 128}
# hidden-size sweep of the LSTM and Transformer cores (hidden 128 is the core comparison's):
# (hidden size, learning rate of the abstract model, learning rate of the natural model)
HIDDEN_SWEEP = [(64, '3e-3', '3e-4'), (256, '3e-3', '3e-4'), (512, '3e-4', '1e-4'), (1024, '3e-4', '1e-4')]

AB_DIR = CHECKPOINT_DIR / 'abstract'
# reruns of the abstract training runs that keep saving checkpoints until validation accuracy reaches
# 0.80 (train_abstract.py --save_until_acc 0.8), for the checkpoints matched at 0.80
AB_U80_DIR = CHECKPOINT_DIR / 'abstract_until080'
NAT_DIR = CHECKPOINT_DIR / 'natural'
DIM_MATCH_DIR = CHECKPOINT_DIR / 'dim_match'


def _ab(name, rnn, hs, acc_match=None, any_epoch=False, root=AB_DIR):
    cfg = {'folder': f'{root}/{name}', 'file': 'final_RNN_model', 'hs': hs, 'rnn': rnn,
           'match_acc': acc_match is not None, 'acc_match': acc_match}
    if any_epoch:
        cfg['any_epoch'] = True
    return cfg


# One-hot abstract models. Key suffix: none = epoch-0 checkpoint closest to 0.75 validation accuracy,
# _U80 = checkpoint closest to 0.80 at any epoch, _FINAL = final checkpoint.
AB_CONFIG_DICT = {}
for rnn in CORES:
    name = f'hs{HIDDEN_DICT[rnn]}_{rnn.lower()}_lr3e-3_wd1e-4_nomturk'
    AB_CONFIG_DICT[f'AB_{rnn}_NOMTURK'] = _ab(name, rnn, HIDDEN_DICT[rnn], DEFAULT_MATCH_ACC)
    AB_CONFIG_DICT[f'AB_{rnn}_NOMTURK_FINAL'] = _ab(name, rnn, HIDDEN_DICT[rnn])
    AB_CONFIG_DICT[f'AB_{rnn}_NOMTURK_U80'] = _ab(name, rnn, HIDDEN_DICT[rnn], 0.80, True, AB_U80_DIR)
for rnn in ['LSTM', 'DECODER']:
    for hs, lr, _ in HIDDEN_SWEEP:
        name = f'hs{hs}_{rnn.lower()}_lr{lr}_wd1e-4_nomturk'
        AB_CONFIG_DICT[f'AB_{rnn}_{hs}_NOMTURK_FINAL'] = _ab(name, rnn, hs)
        AB_CONFIG_DICT[f'AB_{rnn}_{hs}_NOMTURK_U80'] = _ab(name, rnn, hs, 0.80, True, AB_U80_DIR)


def _nat(bb, rnn, hs=128, lr='3e-4', name=None, **extra):
    backbone = BACKBONES[bb]
    name = name or f'{backbone}_hs{hs}_{rnn.lower()}_lr{lr}'
    return {'folder': f'{NAT_DIR}/{name}', 'suffix': '_wd1e-4_nomturk', 'file': 'final_CNNRNN_model',
            'backbone': backbone, 'rnn': rnn, 'hs': hs, 'match_acc': False, 'acc_match': None, **extra}


# Natural models, final checkpoints, named <encoder>_<core>[_<hidden size>]_NOMTURK.
NAT_CONFIG_DICT = {}
# the five sequence cores on ResNet-50 CLIP features
for rnn in CORES:
    NAT_CONFIG_DICT[f'ResNet_CLIP_{rnn}_NOMTURK'] = _nat('ResNet_CLIP', rnn, HIDDEN_DICT[rnn])
# vision encoder comparison: LSTM and Transformer cores on the other base-size encoders
for bb in ['ResNet_IN1k', 'ViT_IN21k', 'ViT_CLIP', 'ViT_DINO', 'ViT_DINOv3', 'EVA02_IN22k', 'EVA02_CLIP']:
    for rnn in ['LSTM', 'DECODER']:
        NAT_CONFIG_DICT[f'{bb}_{rnn}_NOMTURK'] = _nat(bb, rnn)
# encoder size: ConvNeXt-IN22k tiny to xlarge and EVA02-CLIP large and enormous, LSTM core
for bb in [f'ConvNeXt_{size}' for size in ['TINY', 'SMALL', 'BASE', 'LARGE', 'XLARGE']] + ['EVA02_LARGE',
                                                                                         'EVA02_ENORMOUS']:
    NAT_CONFIG_DICT[f'{bb}_LSTM_NOMTURK'] = _nat(bb, 'LSTM')
# hidden size, on ResNet-50 CLIP features
for rnn in ['LSTM', 'DECODER']:
    for hs, _, lr in HIDDEN_SWEEP:
        NAT_CONFIG_DICT[f'ResNet_CLIP_{rnn}_{hs}_NOMTURK'] = _nat('ResNet_CLIP', rnn, hs, lr)
# random encoder: one frozen, randomly initialized ResNet-50 (seed 2025) shared by all six seeds, whose
# features have their own cache
NAT_CONFIG_DICT['ResNet_CLIP_LSTM_RANDVIS_H8_NOMTURK'] = _nat(
    'ResNet_CLIP', 'LSTM', name=f"{BACKBONES['ResNet_CLIP']}_randinit_hs128_lstm_lr3e-4",
    emb_cache_dir=str(FEATURE_DIR / 'randinit_seed2025'))

# the stimulus part of the abstract input, without the task one-hot (task/base.py embedding_size)
DIM_MATCH_EMBEDDING_SIZE = INPUT_SIZE - len(TASK_NAME_TASK_INDEX)
# its last 4 channels are continuous (3 view angles and size); they set the pose-dependent noise
DIM_MATCH_N_CONT = 4


def _dm(noise_scale, noise_pose_k, acc_match=None):
    return {'folder': f'{DIM_MATCH_DIR}/hs128_lstm_lr3e-4_wd1e-4_dim2048_ns{noise_scale}_k{noise_pose_k}_nomturk',
            'file': 'final_RNN_model', 'hs': 128, 'rnn': 'LSTM',
            'match_acc': acc_match is not None, 'acc_match': acc_match,
            'upsample_dim': 2048, 'num_tokens': 49,
            'noise_scale': float(noise_scale), 'noise_pose_k': float(noise_pose_k)}


# Dimensionality-matched LSTM: the abstract input randomly projected to 49 tokens x 2048 channels (the
# ResNet-50 feature shape) plus Gaussian noise of level s scaled by pose (k = 0.5). DIM_MATCH_LSTM_NOMTURK
# (s = 0.35) is the one used in the substitutions. Final checkpoints, except the noise-free model at the
# epoch-0 checkpoint closest to 0.75.
DIM_MATCH_CONFIG_DICT = {
    'DIM_MATCH_LSTM_NOMTURK': _dm('0.35', '0.5'),
    'DIM_MATCH_LSTM_NONOISE_NOMTURK': _dm('0.0', '0.0', DEFAULT_MATCH_ACC),
    'DIM_MATCH_LSTM_NONOISE_NOMTURK_FINAL': _dm('0.0', '0.0'),
    **{f'DIM_MATCH_LSTM_NS{s}_NOMTURK': _dm(s, '0.5') for s in ['0.25', '0.4', '0.45', '0.5']},
}


def load_all_experiment_models(
        n_models=6,
        ab_model_keys=None,
        nat_model_keys=None,
        dim_match_model_keys=None,
        custom_ab_configs=None,
        custom_nat_configs=None,
        custom_dim_match_configs=None,
        load_random=False,
):
    """Trained models of the given keys, `n_models` seeds each, in eval mode.

    Returns ``(natural, abstract)``, each ``{key: [model of seed 0, 1, ...]}``. The dimensionality-matched
    models are in the abstract dict, since they take the same one-hot input. With ``load_random`` every
    key also gets ``'<key>_rand'``: the same architectures at their random initialization.
    """
    ab_models = _load_model_set(ab_model_keys, {**AB_CONFIG_DICT, **(custom_ab_configs or {})},
                                n_models, 'abstract', load_random)
    nat_models = _load_model_set(nat_model_keys, {**NAT_CONFIG_DICT, **(custom_nat_configs or {})},
                                 n_models, 'natural', load_random)
    ab_models.update(_load_model_set(
        dim_match_model_keys, {**DIM_MATCH_CONFIG_DICT, **(custom_dim_match_configs or {})},
        n_models, 'dim_match', load_random))
    return nat_models, ab_models


def _load_model_set(keys, configs, n_models, family, load_random):
    init_fn = {'abstract': init_ab, 'natural': init_nat, 'dim_match': init_dim_match}[family]
    models = {}
    for key in keys or []:
        cfg = configs[key]
        params = _get_init_params(cfg, family)
        trained, random_init = [], []
        for rep in range(n_models):
            if load_random:
                random_init.append(init_fn(seed=2025 + rep, **params))
            model = init_fn(seed=2025 + rep, **params)
            model.load_state_dict(load_weights(rep, cfg, drop_backbone=family == 'natural'))
            trained.append(model)
        models[key] = trained
        if load_random:
            models[f'{key}_rand'] = random_init
        logger.info(f'[{family}] loaded {len(trained)} seeds of {key}')
    return models


def _get_init_params(cfg, family):
    """Arguments of the family's init function for a config."""
    params = {'rnn_type': cfg['rnn'], 'hidden': cfg['hs']}
    if family == 'natural':
        params['num_tokens'], params['feat_dim'] = cached_token_shape(cfg)
    elif family == 'dim_match':
        params.update({k: cfg[k] for k in ('upsample_dim', 'num_tokens', 'noise_scale', 'noise_pose_k')})
    return params


def run_dir(cfg, rep):
    """Run folder of seed `rep` of a config."""
    return Path(f"{cfg['folder']}_rep{rep}{cfg.get('suffix', '')}")


def find_best_checkpoint(
        parent_dir: Path,
        file_prefix: str,
        seed: int,
        match_acc: bool = True,
        acc_match: float = 0.70,
        any_epoch: bool = False,
) -> Path:
    """The checkpoint in `parent_dir` with validation accuracy closest to `acc_match`, or the final one.

    Accuracy matching searches the mid-epoch checkpoints of epoch 0
    (``RNN_model_acc0.74_epoch_0_batch_1500.pth``), or of every epoch with ``any_epoch``; a tie goes to
    the first file in glob order. Without matching, or without mid-epoch checkpoints, it returns the
    final checkpoint ``{file_prefix}_rep{seed}.pth``.
    """
    if match_acc:
        assert acc_match is not None, 'provide matched accuracy'
    batch_pattern = "RNN_model_acc*_epoch_*_batch_*.pth" if any_epoch else "RNN_model_acc*_epoch_0_batch_*.pth"
    batch_files = list(parent_dir.glob(batch_pattern))

    if batch_files and match_acc:
        def get_acc_diff(p):
            try:
                # 'RNN_model_acc0.68_epoch_0_batch_10.pth' -> 0.68
                acc_str = p.stem.split('acc')[-1].split('_epoch')[0]
                return abs(float(acc_str) - acc_match)
            except (ValueError, IndexError):
                return float('inf')

        target = min(batch_files, key=get_acc_diff)
        logger.info(f"Selected checkpoint closest to {acc_match}: {target.name}")
        return target

    fallback = parent_dir / f"{file_prefix}_rep{seed}.pth"
    logger.info(f"Selected checkpoint: {fallback}")
    if not fallback.is_file():
        raise FileNotFoundError(f"No early batches and no final model at {fallback}")
    return fallback


def load_weights(rep, cfg, drop_backbone=False):
    """State dict of the checkpoint a config selects for seed `rep`.

    ``drop_backbone`` removes the frozen vision encoder (``vision_backbone.*``) that a natural checkpoint
    may hold; evaluation reads the encoder's features from the cache instead. The ``_orig_mod.`` prefix
    of a checkpoint saved from a ``torch.compile``-d model is removed.
    """
    parent_dir = run_dir(cfg, rep)
    if not parent_dir.is_dir():
        raise FileNotFoundError(f"Model directory missing: {parent_dir}")
    target_path = find_best_checkpoint(
        parent_dir, cfg['file'], rep,
        match_acc=cfg['match_acc'],
        acc_match=cfg['acc_match'],
        any_epoch=cfg.get('any_epoch', False),
    )
    state_dict = load_repair_checkpoint(str(target_path))
    if drop_backbone:
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vision_backbone.')}
    return state_dict


def _emb_cache_paths(cfg):
    paths = find_emb_cache(cfg['backbone'], cfg.get('emb_cache_dir'))
    if not paths:
        raise FileNotFoundError(f"No cached {cfg['backbone']} features in {cfg.get('emb_cache_dir') or FEATURE_DIR}; "
                                f"run cache_image_embed.py first")
    return paths


def load_emb_cache(keys, configs=None):
    """Cached encoder features of natural configs, ``{key: (features, blank)}``.

    ``features`` holds one frame per HvM image and ``blank`` the frame of a blank (delay) screen, as
    cache_image_embed.py wrote them for the config's encoder.
    """
    active_nat = {**NAT_CONFIG_DICT, **(configs or {})}
    emb_cache_dict = dict()
    for k in keys:
        paths = _emb_cache_paths(active_nat[k])
        emb_cache_dict[k] = (torch.load(paths[0]), torch.load(paths[2]))
    return emb_cache_dict


def cached_token_shape(cfg):
    """``(num_tokens, feat_dim)`` of one frame of a natural config's cached features."""
    blank = torch.load(_emb_cache_paths(cfg)[2])
    return tuple(cached_feat_to_tokens(blank[0]).shape)


def init_nat(rnn_type, hidden, num_tokens, feat_dim, seed=2025):
    """A :class:`CNNRNN` without its vision encoder, for cached features of ``(num_tokens, feat_dim)`` per frame.

    The encoder stays frozen in training, so this model holds every trained weight.
    """
    seed_everything(seed)
    model = CNNRNN(
        hidden_size=hidden,
        output_size=2,
        label_size=len(TASK_NAME_TASK_INDEX),
        seq_model_type=rnn_type,
        dropout=0.3,
        num_heads=NUM_HEADS,
        no_backbone=True,
        feat_dim=feat_dim,
        num_input_tokens=num_tokens,
    )
    model.eval()
    return model


def init_ab(rnn_type, hidden=128, seed=2025):
    seed_everything(seed)
    model = CustomRNN(
        input_size=INPUT_SIZE,
        hidden_size=hidden,
        output_size=2,
        seq_model_type=rnn_type,
        num_heads=NUM_HEADS,
    )
    model.eval()
    return model


def init_dim_match(rnn_type, hidden=128, seed=2025, upsample_dim=2048, num_tokens=49,
                   noise_scale=0.0, noise_pose_k=0.0):
    """A :class:`DimMatchCNNRNN` built as train_abstract_dim_match.py builds it.

    The projection is a buffer, so a trained model gets the checkpoint's; the one drawn here (seeded as
    in training) only matters for the random-init twin.
    """
    seed_everything(seed)
    core = CNNRNN(
        hidden_size=hidden,
        output_size=2,
        label_size=len(TASK_NAME_TASK_INDEX),
        seq_model_type=rnn_type,
        dropout=0.3,
        num_heads=NUM_HEADS,
        no_backbone=True,
        feat_dim=upsample_dim,
        num_input_tokens=num_tokens,
    )
    projection = build_projection(DIM_MATCH_EMBEDDING_SIZE, num_tokens, upsample_dim, seed)
    model = DimMatchCNNRNN(
        core=core,
        projection=projection,
        n_cont=DIM_MATCH_N_CONT,
        noise_scale=noise_scale,
        noise_pose_k=noise_pose_k,
    )
    model.eval()
    return model
