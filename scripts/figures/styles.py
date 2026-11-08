"""Model lists and plot styles of the figures.

A bar style is the dict analysis/plot.py reads: colour, hatch, the marker drawn above the bar, and
`group`, the bracket labels under the x-axis from the outer to the inner level. Lists set the plot
order. Keys are the evaluation config keys of analysis/factory.py: `AB_*` are one-hot abstract models
(`_U80`: the 80%-matched checkpoint, whose validation accuracy is closest to 0.80; `_FINAL`: the
final checkpoint; no suffix: 75%-matched); the other keys are natural models named after their
vision encoder (ResNet_CLIP = ResNet-50 CLIP) and sequence core.
"""
from utils.const import VLM_MODELS

CORES = ['RNN', 'GRU', 'LSTM', 'BERT', 'DECODER']
CORE_MARKER = {'RNN': 'o', 'GRU': '^', 'LSTM': 's', 'BERT': 'X', 'DECODER': 'P'}

# sequence-core comparison
SEQ_TYPE_KEYS = [f'AB_{c}_NOMTURK_U80' for c in CORES] + [f'ResNet_CLIP_{c}_NOMTURK' for c in CORES]
AB_CORE_COLOR = {'RNN': '#e6550d', 'GRU': '#fd8d3c', 'LSTM': '#fdae6b', 'BERT': '#fdd0a2', 'DECODER': '#a63603'}
NAT_CORE_COLOR = {'RNN': '#3182bd', 'GRU': '#6baed6', 'LSTM': '#9ecae1', 'BERT': '#756bb1', 'DECODER': '#bcbddc'}
SEQ_TYPE_STYLES = {
    **{f'AB_{c}_NOMTURK_U80': {'color': AB_CORE_COLOR[c], 'group': ['Abstract'], 'marker': CORE_MARKER[c], 'hatch': ''}
       for c in CORES},
    **{f'ResNet_CLIP_{c}_NOMTURK': {'color': NAT_CORE_COLOR[c], 'group': ['Natural'], 'marker': CORE_MARKER[c],
                                    'hatch': ''} for c in CORES},
    'nemotron-nano-12b-v2-vl': {'color': '#31a354', 'group': ['VLM'], 'marker': 'D', 'hatch': ''},
    'nemotron-nano-12b-v2-vl-_no_think': {'color': '#74c476', 'group': ['VLM'], 'marker': 'D', 'hatch': '//'},
    'qwen3-vl-30b-a3b-thinking': {'color': '#a1d99b', 'group': ['VLM'], 'marker': 'D', 'hatch': 'xx'},
    'qwen3-vl-235b-a22b-thinking': {'color': '#c7e9c0', 'group': ['VLM'], 'marker': 'D', 'hatch': '..'},
    'gemini-3.1-pro-preview': {'color': '#e5f5e0', 'group': ['VLM'], 'marker': 'D', 'hatch': '\\\\'},
    'human': {'color': '#333333', 'group': ['Human'], 'hatch': '..'},
}
assert set(VLM_MODELS) <= set(SEQ_TYPE_STYLES)

# the abstract cores at three checkpoints next to the natural ResNet-CLIP LSTM
AB_MATCH_LEVELS = [('', '75%', ''), ('_U80', '80%', '\\'), ('_FINAL', 'Final', '/')]  # key suffix, bracket, hatch
AB_MATCH_LEVEL_KEYS = [f'AB_{c}_NOMTURK{sfx}' for sfx, _, _ in AB_MATCH_LEVELS for c in CORES] + [
    'ResNet_CLIP_LSTM_NOMTURK']
AB_MATCH_LEVEL_STYLES = {
    **{f'AB_{c}_NOMTURK{sfx}': {'color': AB_CORE_COLOR[c], 'group': ['Abstract', label], 'marker': CORE_MARKER[c],
                                'hatch': hatch}
       for sfx, label, hatch in AB_MATCH_LEVELS for c in CORES},
    'ResNet_CLIP_LSTM_NOMTURK': {'color': NAT_CORE_COLOR['LSTM'], 'group': [], 'marker': CORE_MARKER['LSTM'],
                                 'hatch': ''},
    'human': {'color': '#333333', 'group': ['Human'], 'hatch': '..'},
}

# vision encoders: colour = encoder architecture, (LSTM, Transformer) = (darker, lighter) step of it,
# hatch = pretraining. The lighter fills are below 3:1 contrast on white, so every bar keeps its
# black outline, hatch and marker.
VISION_ARCHS = {  # key prefix -> (legend label, bracket label, (LSTM colour, Transformer colour))
    'ResNet': ('ResNet-50', 'ResNet', ('#1666aa', '#60aaf3')),
    'ViT': ('ViT-B/16', 'ViT', ('#357426', '#77b868')),
    'EVA02': ('EVA02-B', 'EVA', ('#834791', '#c889d7')),
}
VISION_PRETRAIN = {  # key infix -> (legend label, hatch)
    'IN1k': ('Supervised ImageNet', ''), 'IN21k': ('Supervised ImageNet', ''),
    'CLIP': ('CLIP', '//'), 'DINO': ('DINO', '\\\\'), 'DINOv3': ('DINOv3', 'xx'), 'IN22k': ('MIM', '--'),
}
VISION_NAT_KEYS = [
    'ResNet_IN1k_LSTM_NOMTURK', 'ResNet_CLIP_LSTM_NOMTURK',
    'ResNet_IN1k_DECODER_NOMTURK', 'ResNet_CLIP_DECODER_NOMTURK',
    'ViT_IN21k_LSTM_NOMTURK', 'ViT_CLIP_LSTM_NOMTURK', 'ViT_IN21k_DECODER_NOMTURK', 'ViT_CLIP_DECODER_NOMTURK',
    'ViT_DINO_LSTM_NOMTURK', 'ViT_DINOv3_LSTM_NOMTURK', 'ViT_DINO_DECODER_NOMTURK', 'ViT_DINOv3_DECODER_NOMTURK',
    'EVA02_IN22k_LSTM_NOMTURK', 'EVA02_CLIP_LSTM_NOMTURK',
    'EVA02_IN22k_DECODER_NOMTURK', 'EVA02_CLIP_DECODER_NOMTURK',
]
VISION_TYPE_KEYS = ['AB_LSTM_NOMTURK_U80', 'AB_DECODER_NOMTURK_U80'] + VISION_NAT_KEYS


def _vision_style(key):
    arch, pretrain, core, _ = key.split('_')
    _, bracket, colors = VISION_ARCHS[arch]
    return {'color': colors[['LSTM', 'DECODER'].index(core)], 'group': ['Natural', bracket],
            'marker': CORE_MARKER[core], 'hatch': VISION_PRETRAIN[pretrain][1]}


VISION_TYPE_STYLES = {
    'AB_LSTM_NOMTURK_U80': {'color': '#e6550d', 'group': ['Abstract'], 'marker': 's', 'hatch': ''},
    'AB_DECODER_NOMTURK_U80': {'color': '#fd8d3c', 'group': ['Abstract'], 'marker': 'P', 'hatch': ''},
    **{k: _vision_style(k) for k in VISION_NAT_KEYS},
    'human': {'color': '#333333', 'group': ['Human'], 'hatch': '..'},
}

# hidden size of the LSTM and Transformer cores (hidden 128 is the seq_type model), one line per family
HIDDEN_SIZES = [64, 128, 256, 512, 1024]


def _sized(prefix, core, suffix):
    return {h: f"{prefix}{core}{'' if h == 128 else f'_{h}'}_NOMTURK{suffix}" for h in HIDDEN_SIZES}


SEQ_SIZE_FAMILIES = {  # line label -> (colour, marker, {hidden size: key})
    'Abstract LSTM': ('#e6550d', 's', _sized('AB_', 'LSTM', '_U80')),
    'Abstract Transformer': ('#de2d26', 'P', _sized('AB_', 'DECODER', '_U80')),
    'Natural LSTM': ('#3182bd', 's', _sized('ResNet_CLIP_', 'LSTM', '')),
    'Natural Transformer': ('#756bb1', 'P', _sized('ResNet_CLIP_', 'DECODER', '')),
}
SEQ_SIZE_KEYS = [k for _, _, keys in SEQ_SIZE_FAMILIES.values() for k in keys.values()]

# vision encoder size, LSTM core, final checkpoints: line label -> (colour, {encoder: model key})
ENCODER_SIZE_FAMILIES = {
    'ConvNeXt-IN22k LSTM': ('#1b9e77', {f'ConvNeXt_{s}': f'ConvNeXt_{s}_LSTM_NOMTURK'
                                        for s in ['TINY', 'SMALL', 'BASE', 'LARGE', 'XLARGE']}),
    'EVA02-CLIP LSTM': (VISION_ARCHS['EVA02'][2][0], {f'EVA02_{s}': f'EVA02_{s}_LSTM_NOMTURK'
                                                      for s in ['CLIP', 'LARGE', 'ENORMOUS']}),
}

# metric alignment: the seq_type, seq_size and vision_type models plus the EVA02-CLIP L and E LSTMs,
# one point each, coloured by encoder family with the core's marker
METRIC_ALIGNMENT_KEYS = (
    SEQ_TYPE_KEYS[:5]
    + [f'AB_{c}_{h}_NOMTURK_U80' for c in ['LSTM', 'DECODER'] for h in [64, 256, 512, 1024]]
    + SEQ_TYPE_KEYS[5:]
    + [k for k in VISION_NAT_KEYS if not k.startswith('ResNet_CLIP_')]
    + ['EVA02_LARGE_LSTM_NOMTURK', 'EVA02_ENORMOUS_LSTM_NOMTURK']
    + [f'ResNet_CLIP_{c}_{h}_NOMTURK' for c in ['LSTM', 'DECODER'] for h in [64, 256, 512, 1024]]
)
FAMILY_COLOR = {'Abstract': '#ff7f0e', 'ResNet': '#1f77b4', 'ViT': '#2ca02c', 'EVA': '#9467bd'}


def family(key):
    """Encoder family of a model key (Abstract for the one-hot models)."""
    for prefix, name in [('EVA02_', 'EVA'), ('ViT_', 'ViT'), ('ResNet_', 'ResNet')]:
        if key.startswith(prefix):
            return name
    return 'Abstract'


# the encoders of the separability panels, each against the LSTM trained on it
SEPARABILITY_ENCODERS = ['ResNet_IN1k', 'ResNet_CLIP', 'ViT_IN21k', 'ViT_CLIP', 'ViT_DINO', 'ViT_DINOv3',
                         'EVA02_IN22k', 'EVA02_CLIP', 'EVA02_LARGE', 'EVA02_ENORMOUS']

# substitutions: colour = recipient, dark to light as less of the recipient retrains; marker = condition
CONDITIONS = ['native', 'theta', 'theta_wout', 'theta_win', 'untrained']
COND_MARKER = {'native': 'o', 'theta': '^', 'theta_wout': 's', 'theta_win': 'D', 'untrained': 'X'}
COND_LABEL = {
    'native': 'native',
    'theta': r'$\theta$',
    'theta_wout': r'$\theta{+}W_{out}$',
    'theta_win': r'$\theta{+}W_{in}$',
    'untrained': r'untrained $\theta$',
}
RECIPIENT_NAME = {'abs': 'Abstract', 'nat': 'Natural'}
RECIPIENT_COLORS = {
    'abs': ['#a63603', '#e6550d', '#fd8d3c', '#fdbe85', '#fdd0a2'],
    'nat': ['#08519c', '#3182bd', '#4292c6', '#6baed6', '#9ecae1'],
}


def cond_color(recipient, cond):
    return RECIPIENT_COLORS[recipient][CONDITIONS.index(cond)]


def substitution_order(keys_by_recipient):
    """Plot order of a substitution set {recipient: {condition: key}}."""
    return [keys[c] for keys in keys_by_recipient.values() for c in CONDITIONS if c in keys]


def substitution_styles(keys_by_recipient):
    """Bar styles of a substitution set, plus the human bar."""
    out = {key: {'color': cond_color(rec, cond), 'group': [RECIPIENT_NAME[rec]], 'marker': COND_MARKER[cond],
                 'hatch': ''}
           for rec, keys in keys_by_recipient.items() for cond, key in keys.items()}
    out['human'] = {'color': '#333333', 'group': ['Human'], 'hatch': ''}
    return out
