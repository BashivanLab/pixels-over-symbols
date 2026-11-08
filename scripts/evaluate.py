"""Evaluate the models of the paper on the MTurk trials and cache their responses in data/evaluations.

usage: python scripts/evaluate.py GROUP [GROUP ...] [--features] [--device DEVICE]
       python scripts/evaluate.py --list

Each group is one set of models of the paper (six seeds per model):
    vlm             the five VLMs
    abstract        one-hot abstract models, five sequence cores, at the epoch-0 checkpoint closest to 0.75
                    validation accuracy, the checkpoint closest to 0.80, and the final checkpoint
    natural         natural models, five sequence cores on ResNet-50 CLIP, and the ViT-B/16 CLIP LSTM
    backbone        LSTM and Transformer cores on the eight vision encoders of the encoder comparison
                    (ResNet-50, ViT-B/16 and EVA02-B with different pretraining)
    encoder_size    LSTM on ConvNeXt-IN22k tiny to xlarge and on EVA02-CLIP base, large and enormous
    hidden_size     LSTM and Transformer cores of hidden size 64 to 1024, one-hot abstract (0.80-matched
                    and final) and natural
    random_encoder  LSTM on a frozen random ResNet-50, and on the pretrained one
    dim_match       dimensionality-matched LSTM at every noise level, and the random-init twin of each
    substitution    the Abs->Nat and Nat->Abs substitutions and the native models they come from, which
                    are also the systems of the DSA

A model whose responses are cached already is not evaluated again, so a group can be rerun to fill in
what is missing. The participants' responses (human_responses.npy, feature_indices.npy, pair_keys.npy)
are written with the first group. Every group then prints the accuracy and the human consistency
rho(m, h) of its models and writes all metrics to data/evaluations/summary_<group>.csv. With --features
the hidden states the DSA reads (scripts/dsa/run_dsa.py) are cached too.

Natural models read their vision encoder's features from data/features (cache_image_embed.py) and
the encoder itself is never run; a group keeps the feature caches of its natural models in memory
together, about 15 GB for encoder_size. The paper's responses were computed on CPU, as by default here.

The cached responses do not depend on PYTHONHASHSEED. The cross-task and cross-stimulus RSA computed
from them does (see analysis.experiment_pipeline.run_experiment_pipeline): the figure scripts run with
PYTHONHASHSEED=0.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.experiment_pipeline import embedding_pipeline, run_experiment_pipeline  # noqa: E402
from analysis.factory import AB_CONFIG_DICT, CORES, DIM_MATCH_CONFIG_DICT, NAT_CONFIG_DICT  # noqa: E402
from analysis.summary import summarize_pipeline  # noqa: E402
from analysis.transplant_configs import AB_CONFIGS, ARMS, DM_CONFIGS, NAT_CONFIGS  # noqa: E402
from utils.const import VLM_MODELS  # noqa: E402
from utils.paths import EVAL_DIR  # noqa: E402

ALL_AB = {**AB_CONFIG_DICT, **AB_CONFIGS}
ALL_NAT = {**NAT_CONFIG_DICT, **NAT_CONFIGS}
ALL_DM = {**DIM_MATCH_CONFIG_DICT, **DM_CONFIGS}

ENCODERS = ['ResNet_IN1k', 'ResNet_CLIP', 'ViT_IN21k', 'ViT_CLIP', 'ViT_DINO', 'ViT_DINOv3', 'EVA02_IN22k', 'EVA02_CLIP']
SIZES = [64, 128, 256, 512, 1024]
GROUPS = {
    'vlm': [],
    'abstract': [f'AB_{c}_NOMTURK{sfx}' for sfx in ['', '_U80', '_FINAL'] for c in CORES],
    'natural': [f'ResNet_CLIP_{c}_NOMTURK' for c in CORES] + ['ViT_CLIP_LSTM_NOMTURK'],
    'backbone': [f'{bb}_{c}_NOMTURK' for bb in ENCODERS for c in ['LSTM', 'DECODER']],
    'encoder_size': ([f'ConvNeXt_{s}_LSTM_NOMTURK' for s in ['TINY', 'SMALL', 'BASE', 'LARGE', 'XLARGE']]
                     + [f'EVA02_{s}_LSTM_NOMTURK' for s in ['CLIP', 'LARGE', 'ENORMOUS']]),
    'hidden_size': ([f"AB_{c}{'' if hs == 128 else f'_{hs}'}_NOMTURK{sfx}"
                     for c in ['LSTM', 'DECODER'] for hs in SIZES for sfx in ['_U80', '_FINAL']]
                    + [f"ResNet_CLIP_{c}{'' if hs == 128 else f'_{hs}'}_NOMTURK"
                       for c in ['LSTM', 'DECODER'] for hs in SIZES]),
    'random_encoder': ['ResNet_CLIP_LSTM_RANDVIS_H8_NOMTURK', 'ResNet_CLIP_LSTM_NOMTURK'],
    'dim_match': list(DIM_MATCH_CONFIG_DICT),
    'substitution': list(dict.fromkeys(k for arm in ARMS.values() for k in arm.values())),
}
# groups whose figures also show the random-init twin of every model
WITH_RANDOM_TWINS = {'dim_match'}


def evaluate(group, features=False, device='cpu'):
    keys = GROUPS[group]
    ab = [k for k in keys if k in ALL_AB]
    nat = [k for k in keys if k in ALL_NAT]
    dm = [k for k in keys if k in ALL_DM]
    assert len(ab) + len(nat) + len(dm) == len(keys), set(keys) - set(ab + nat + dm)
    vlm = VLM_MODELS if group == 'vlm' else None
    data = run_experiment_pipeline(
        group, ab_model_keys=ab, nat_model_keys=nat, dim_match_model_keys=dm, vlm_model_keys=vlm,
        custom_ab_configs=AB_CONFIGS, custom_nat_configs=NAT_CONFIGS, custom_dim_match_configs=DM_CONFIGS,
        device=device, load_random=group in WITH_RANDOM_TWINS)

    responders = keys + (vlm or [])
    df = summarize_pipeline(data, groups={group: responders})
    df.to_csv(EVAL_DIR / f'summary_{group}.csv', index=False)
    acc, cons = data['acc_result']['overall'], data['cons_result']['overall']
    print(f"\n{'model':40s} {'accuracy':>15s} {'rho(m,h)':>16s}")
    for k in [name for key in responders for name in (key, f'{key}_rand') if name in acc['models']]:
        a, mh = acc['models'][k], cons['models'][k]['mh']
        print(f"{k:40s} {a['mean']:.3f} +/- {a['std']:.3f}  {mh['mean']:+.3f} +/- {mh['std']:.3f}")
    print(f"{'human':40s} {acc['human']['mean']:.3f} +/- {acc['human']['std']:.3f}")

    if features and keys:
        embedding_pipeline(
            group, ab_model_keys=ab, nat_model_keys=nat, dim_match_model_keys=dm,
            custom_ab_configs=AB_CONFIGS, custom_nat_configs=NAT_CONFIGS, custom_dim_match_configs=DM_CONFIGS,
            device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('groups', nargs='*', metavar='GROUP', help=f"one or more of: {', '.join(GROUPS)}")
    parser.add_argument('--features', action='store_true',
                        help='also cache the hidden states of the models (sequence-core input and output)')
    parser.add_argument('--device', default='cpu', help='torch device of the evaluation')
    parser.add_argument('--list', action='store_true', help='print the model keys of every group')
    args = parser.parse_args()
    if args.list:
        for name, keys in GROUPS.items():
            print(f"{name}: {' '.join(keys or VLM_MODELS)}")
        return
    unknown = [g for g in args.groups if g not in GROUPS]
    if unknown or not args.groups:
        parser.error(f"name one or more groups of: {', '.join(GROUPS)}")
    for group in args.groups:
        evaluate(group, args.features, args.device)


if __name__ == '__main__':
    main()
