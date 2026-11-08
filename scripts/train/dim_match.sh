#!/bin/bash
# Dimensionality-matched Abstract models (LSTM, hidden 128) at noise level s.
#
# Usage: scripts/train/dim_match.sh <noise_scale> <seed>
#   noise_scale: s = 0, 0.25, 0.35 (the main dimensionality-matched model), 0.4, 0.45 or 0.5.
#                s = 0 has no noise and keeps its epoch-0 checkpoints with validation
#                accuracy between 0.6 and 0.8; the noisy models scale the noise with
#                pose extremity (noise_pose_k 0.5).
#   seed:        0-5 (the run and the random projection are seeded with 2025 + seed)
#
# Most of the paper's dimensionality-matched models were trained on the CPU (s = 0.35 on a
# GPU), so this script hides the GPUs; weights can differ bitwise between devices.
set -euo pipefail
cd "$(dirname "$0")/../.."

S=$1; SEED=$2
if [ "$S" = 0 ] || [ "$S" = 0.0 ]; then
  S=0.0; K=0.0; EXTRA="--save_mid_epoch"
else
  K=0.5; EXTRA=""
fi
OUT=data/checkpoints/dim_match/hs128_lstm_lr3e-4_wd1e-4_dim2048_ns${S}_k${K}_nomturk_rep${SEED}
mkdir -p "$OUT"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} \
python train_abstract_dim_match.py \
  --batch_size 128 --train_dataset_size 256000 --val_dataset_size 64000 --num_epochs 10 \
  --lr 3e-4 --weight_decay 1e-4 --dropout 0.3 --grad_norm 2.0 --warmup_epochs 0 \
  --hidden_size 128 --seq_model_type lstm --num_heads 8 --label_smoothing 0.2 \
  --upsample_dim 2048 --noise_scale "$S" --noise_pose_k "$K" \
  --num_workers 6 --pin_mem --rep_index "$SEED" --results_dir "$OUT" $EXTRA \
  2>&1 | tee "$OUT/out.txt"
