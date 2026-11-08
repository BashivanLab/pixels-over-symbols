#!/bin/bash
# Frozen-backbone feature caches of the HvM images (run these before training Natural models).
#
# Usage: scripts/train/features.sh [<cnn_type> ...]
#        scripts/train/features.sh randinit
#   With no argument, caches every backbone of the paper. "randinit" caches the
#   random-encoder model's backbone: a randomly initialized resnet50_clip.openai
#   (seed 2025), written to data/features/randinit_seed2025/ with its weights.
# Uses the GPU in CUDA_VISIBLE_DEVICES (the paper caches were computed on a GPU,
# in float16 autocast); pretrained weights come from the Hugging Face hub cache.
set -euo pipefail
cd "$(dirname "$0")/../.."

BACKBONES=(
  resnet50_clip.openai resnet50.a1_in1k
  vit_base_patch16_clip_224.laion2b_ft_in1k vit_base_patch16_224.orig_in21k
  vit_base_patch16_224.dino vit_base_patch16_dinov3.lvd1689m
  eva02_base_patch14_224.mim_in22k eva02_base_patch16_clip_224.merged2b
  eva02_large_patch14_clip_224.merged2b eva02_enormous_patch14_clip_224.laion2b_plus
  convnext_tiny.fb_in22k convnext_small.fb_in22k convnext_base.fb_in22k
  convnext_large.fb_in22k convnext_xlarge.fb_in22k
)
if [ $# -gt 0 ]; then BACKBONES=("$@"); fi

for bb in "${BACKBONES[@]}"; do
  if [ "$bb" = randinit ]; then
    python cache_image_embed.py --cnn_type resnet50_clip.openai --no_pretrained --seed 2025 --batch_size 32
  else
    python cache_image_embed.py --cnn_type "$bb" --batch_size 32
  fi
done
