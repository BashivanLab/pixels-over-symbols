"""Frozen vision backbones of the Natural models.

Pretrained weights are timm's checkpoints on the Hugging Face hub, fetched through
the standard hub cache, so HF_HOME, HF_HUB_CACHE and HF_HUB_OFFLINE apply.
"""

import logging

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Key prefixes renamed in a hub checkpoint before timm loads it. The EVA02 MIM
# checkpoint stores its final LayerNorm as fc_norm, which only exists with average
# pooling; the headless model (global_pool='') has norm instead, so the paper
# models carry these weights in norm.
RENAMED_KEYS = {
    'eva02_base_patch14_224.mim_in22k': {'fc_norm.': 'norm.'},
}


def _renamed_state_dict(cnn_type):
    """The hub checkpoint of ``cnn_type`` with the key prefixes in RENAMED_KEYS replaced."""
    hub_id = timm.models.get_pretrained_cfg(cnn_type).hf_hub_id
    state_dict = load_file(hf_hub_download(hub_id, 'model.safetensors'))
    renamed = {}
    for k, v in state_dict.items():
        for old, new in RENAMED_KEYS[cnn_type].items():
            if k.startswith(old):
                k = new + k[len(old):]
        renamed[k] = v
    return renamed


def load_vision_backbone(
        cnn_type: str,
        pretrained: bool = True,
        state_dict_path: str = None,
) -> nn.Module:
    """Create a headless (``num_classes=0``, ``global_pool=''``) timm backbone in eval mode.

    Args:
        cnn_type: timm model name with its pretrained tag, e.g. ``'resnet50_clip.openai'``.
        pretrained: Load the pretrained weights from the Hugging Face hub (or its local
            cache). Raises if they cannot be found, rather than falling back to random
            weights.
        state_dict_path: ``torch.save``d backbone ``state_dict`` loaded last,
            independently of ``pretrained``. The random-encoder model uses it to load
            the random draw that ``cache_image_embed.py --no_pretrained`` wrote, so the
            encoder behind the feature cache and the encoder in the checkpoint are the
            same.

    Returns:
        The backbone with its classification head and global pooling removed
        (so ``forward_features`` returns spatial/token features).
    """
    overlay = None
    if pretrained and cnn_type in RENAMED_KEYS:
        overlay = dict(state_dict=_renamed_state_dict(cnn_type))
    model = timm.create_model(
        cnn_type,
        pretrained=pretrained,
        num_classes=0,
        global_pool='',
        pretrained_cfg_overlay=overlay,
    )
    # timm keeps a copy of an overlaid state dict in the config; drop it
    model.pretrained_cfg.pop('state_dict', None)
    logger.info(f"Initialized {cnn_type} with {'pretrained' if pretrained else 'random'} weights.")

    if state_dict_path is not None:
        sd = torch.load(state_dict_path, map_location='cpu')
        model.load_state_dict(sd)
        logger.info(f"Loaded explicit backbone state_dict from {state_dict_path}")

    model.eval()
    return model


def get_pretrained_stats(cnn_type):
    """Return the ``(mean, std)`` image normalization used for the backbone.

    ResNet and ConvNeXt ImageNet weights use the standard ImageNet statistics; every
    CLIP, ViT and EVA02 backbone uses the OpenAI CLIP statistics, including the
    ImageNet-21k and DINO ViTs whose timm configs list other values (the paper models
    were trained this way). Raises ``ValueError`` for anything unrecognized so
    callers cannot silently apply the wrong normalization.
    """
    if (cnn_type.startswith('resnet') and 'clip' not in cnn_type) or cnn_type.startswith('convnext'):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif any(x in cnn_type for x in ['clip', 'vit', 'eva']):
        mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    else:
        logger.error(f"Unsupported model type for stats: {cnn_type}")
        raise ValueError(f'{cnn_type} not recognized')
    return mean, std
