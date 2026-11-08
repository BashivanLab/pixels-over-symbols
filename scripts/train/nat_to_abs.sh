#!/bin/bash
# Nat->Abs substitution: an Abstract model (LSTM, hidden 128) around the frozen core of the Natural model.
#
# Usage: scripts/train/nat_to_abs.sh <recipient> <condition> <seed>
#   recipient: onehot           one-hot Abstract model (data/checkpoints/nat_to_abs)
#              onehot_until080  the same runs stopped at the first checkpoint with validation
#                               accuracy >= 0.80 (data/checkpoints/nat_to_abs_until080)
#              dimmatch         dimensionality-matched Abstract model, s = 0.35
#   condition: theta      frozen core; input and output projections retrain
#              wout       theta + W_out: the donor's output head is frozen too
#              win        theta + W_in: the donor's input pathway is frozen too, only the
#                         output head retrains (dimmatch only)
#              untrained  a frozen, randomly initialized core (no donor)
#   seed:      0-5; the donor is the ResNet-CLIP LSTM Natural model of the same seed
#
# The paper's Nat->Abs models were trained on the CPU (the one-hot ones with 4 threads), so
# this script hides the GPUs; weights can differ bitwise between devices and thread counts.
set -euo pipefail
cd "$(dirname "$0")/../.."

RECIPIENT=$1; COND=$2; SEED=$3
DONOR=data/checkpoints/natural/resnet50_clip.openai_hs128_lstm_lr3e-4_rep${SEED}_wd1e-4_nomturk/final_CNNRNN_model_rep${SEED}.pth
COMMON="--batch_size 128 --train_dataset_size 256000 --val_dataset_size 64000 --num_epochs 10
  --weight_decay 1e-4 --dropout 0.3 --grad_norm 2.0 --warmup_epochs 0 --hidden_size 128
  --seq_model_type lstm --num_heads 8 --num_workers 6 --rep_index $SEED --freeze_seq_model"

case $RECIPIENT in
  onehot|onehot_until080)
    SCRIPT=train_abstract_frozen.py
    ARGS="$COMMON --lr 3e-3 --save_mid_epoch"
    GROUP=nat_to_abs; PRE=hs128_lstm_lr3e-3_wd1e-4_nomturk_frozen
    if [ "$RECIPIENT" = onehot_until080 ]; then GROUP=nat_to_abs_until080; ARGS="$ARGS --save_until_acc 0.8"; fi
    case $COND in
      theta)     NAME=${PRE}_nath8;          ARGS="$ARGS --seq_model_path $DONOR" ;;
      wout)      NAME=${PRE}_nath8_frzfcout; ARGS="$ARGS --seq_model_path $DONOR --freeze_fc_out" ;;
      untrained) NAME=${PRE}_randseq ;;
      *) echo "condition must be theta, wout or untrained" >&2; exit 1 ;;
    esac ;;
  dimmatch)
    SCRIPT=train_abstract_dim_match.py
    ARGS="$COMMON --lr 3e-4 --pin_mem --label_smoothing 0.2 --upsample_dim 2048 --noise_scale 0.35 --noise_pose_k 0.5"
    GROUP=nat_to_abs; PRE=hs128_lstm_lr3e-4_wd1e-4_dim2048_ns0.35_k0.5_nomturk_frozen
    case $COND in
      theta)     NAME=${PRE}_nath8_ls0.2;           ARGS="$ARGS --seq_model_path $DONOR" ;;
      wout)      NAME=${PRE}_nath8_ls0.2_frzfcout;  ARGS="$ARGS --seq_model_path $DONOR --freeze_fc_out" ;;
      win)       NAME=${PRE}_nath8_ls0.2_frzinpath; ARGS="$ARGS --seq_model_path $DONOR --freeze_input_pathway" ;;
      untrained) NAME=${PRE}_nat_ls0.2_randseq ;;
      *) echo "condition must be theta, wout, win or untrained" >&2; exit 1 ;;
    esac ;;
  *) echo "recipient must be onehot, onehot_until080 or dimmatch" >&2; exit 1 ;;
esac
if [ "$COND" != untrained ] && [ ! -f "$DONOR" ]; then echo "missing donor checkpoint $DONOR" >&2; exit 1; fi
OUT=data/checkpoints/$GROUP/${NAME}_rep${SEED}
mkdir -p "$OUT"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} \
python $SCRIPT $ARGS --results_dir "$OUT" 2>&1 | tee "$OUT/out.txt"
