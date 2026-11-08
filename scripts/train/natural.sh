#!/bin/bash
# Natural models: frozen vision backbone + sequence core, trained on the cached backbone features.
#
# Usage: scripts/train/natural.sh <cnn_type> <core> <hidden> <seed> [randinit]
#   cnn_type: timm backbone, e.g. resnet50_clip.openai (the main Natural model); the
#             backbone sweep uses resnet50.a1_in1k, vit_base_patch16_clip_224.laion2b_ft_in1k,
#             vit_base_patch16_224.orig_in21k, vit_base_patch16_224.dino,
#             vit_base_patch16_dinov3.lvd1689m, eva02_base_patch14_224.mim_in22k and
#             eva02_base_patch16_clip_224.merged2b (lstm and decoder); the encoder-size
#             sweep convnext_{tiny,small,base,large,xlarge}.fb_in22k,
#             eva02_large_patch14_clip_224.merged2b and
#             eva02_enormous_patch14_clip_224.laion2b_plus (lstm)
#   core:     rnn | gru | lstm | decoder | bert (with resnet50_clip.openai: rnn at hidden 256,
#             the others at 128)
#   hidden:   128 except as above; the hidden-size sweep trains resnet50_clip.openai lstm and
#             decoder at 64, 256, 512 and 1024
#   seed:     0-5 (the run is seeded with 2025 + seed)
#   randinit: the random-encoder model (resnet50_clip.openai lstm 128): the frozen backbone
#             is the random draw of scripts/train/features.sh randinit
# Run scripts/train/features.sh for the backbone first. Uses the GPU in CUDA_VISIBLE_DEVICES.
set -euo pipefail
cd "$(dirname "$0")/../.."

CNN=$1; CORE=$2; HIDDEN=$3; SEED=$4; MODE=${5:-}
case $HIDDEN in
  512|1024) LR=1e-4 ;;
  *) LR=3e-4 ;;
esac
NAME=${CNN}_hs${HIDDEN}_${CORE}_lr${LR}_rep${SEED}_wd1e-4_nomturk
EXTRA=""
if [ "$MODE" = randinit ]; then
  NAME=${CNN}_randinit_hs${HIDDEN}_${CORE}_lr${LR}_rep${SEED}_wd1e-4_nomturk
  CACHE=data/features/randinit_seed2025
  EXTRA="--no_pretrained --vision_state_dict_path $CACHE/${CNN}_randinit_seed2025.pth --emb_cache_dir $CACHE"
fi
OUT=data/checkpoints/natural/$NAME
mkdir -p "$OUT"

# shellcheck disable=SC2086
python train_natural.py \
  --batch_size 128 --train_dataset_size 256000 --val_dataset_size 64000 --num_epochs 20 \
  --lr "$LR" --weight_decay 1e-4 --dropout 0.3 --grad_norm 2.0 --warmup_epochs 0 \
  --cnn_type "$CNN" --hidden_size "$HIDDEN" --seq_model_type "$CORE" --num_seq_layers 1 \
  --num_heads 8 --train_vars 0 3 6 --val_vars 0 3 6 --num_workers 4 --pin_mem \
  --rep_index "$SEED" --results_dir "$OUT" $EXTRA \
  2>&1 | tee "$OUT/out.txt"
