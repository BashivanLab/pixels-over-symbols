#!/bin/bash
# One-hot Abstract models: the five sequence cores and the hidden-size sweep.
#
# Usage: scripts/train/abstract.sh <core> <hidden> <seed> [until080]
#   core:   rnn | gru | lstm | decoder | bert   (decoder is the Transformer core)
#   hidden: paper models use 256 for rnn and 128 for the others; the hidden-size
#           sweep trains lstm and decoder at 64, 256, 512 and 1024
#   seed:   0-5 (the run is seeded with 2025 + seed)
#   until080: rerun the same trajectory with mid-epoch checkpoints in every epoch
#           and stop at the first one with validation accuracy >= 0.80 (the 0.80
#           accuracy-matched checkpoints), writing to data/checkpoints/abstract_until080
# Without until080 the run writes to data/checkpoints/abstract and saves epoch-0
# checkpoints with validation accuracy between 0.6 and 0.8 (the 0.75-matched ones).
#
# The paper's abstract models were trained on the CPU with 4 threads, so this script
# hides the GPUs; weights can differ bitwise between devices and thread counts.
set -euo pipefail
cd "$(dirname "$0")/../.."

CORE=$1; HIDDEN=$2; SEED=$3; MODE=${4:-}
case $HIDDEN in
  512|1024) LR=3e-4 ;;
  *) LR=3e-3 ;;
esac
GROUP=abstract; EXTRA=""
if [ "$MODE" = until080 ]; then GROUP=abstract_until080; EXTRA="--save_until_acc 0.8"; fi
OUT=data/checkpoints/$GROUP/hs${HIDDEN}_${CORE}_lr${LR}_wd1e-4_nomturk_rep${SEED}
mkdir -p "$OUT"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} \
python train_abstract.py \
  --batch_size 128 --train_dataset_size 256000 --val_dataset_size 64000 --num_epochs 10 \
  --lr "$LR" --weight_decay 1e-4 --dropout 0.3 --grad_norm 2.0 --warmup_epochs 0 \
  --hidden_size "$HIDDEN" --seq_model_type "$CORE" --num_heads 8 --save_mid_epoch \
  --num_workers 6 --rep_index "$SEED" --results_dir "$OUT" $EXTRA \
  2>&1 | tee "$OUT/out.txt"
