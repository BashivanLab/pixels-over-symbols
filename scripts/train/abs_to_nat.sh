#!/bin/bash
# Abs->Nat substitution: a Natural model (LSTM, hidden 128) around the frozen core of an Abstract model.
#
# Usage: scripts/train/abs_to_nat.sh <donor> <condition> <seed> [cnn_type]
#   donor:     final     the final checkpoint of the one-hot Abstract LSTM
#              m80       its 0.80 accuracy-matched checkpoint (abstract_until080)
#              match075  its 0.75 accuracy-matched checkpoint
#              dimmatch  the final dimensionality-matched Abstract LSTM (s = 0.35)
#   condition: theta      frozen core; input and output projections retrain
#              wout       theta + W_out: the donor's output head is frozen too
#              win        theta + W_in: the donor's input pathway is frozen too, only the
#                         output head retrains (dimmatch donor and ResNet-CLIP only)
#              untrained  a frozen, randomly initialized core (the donor is ignored)
#   seed:      0-5; the donor is the Abstract model of the same seed
#   cnn_type:  resnet50_clip.openai (default) or vit_base_patch16_clip_224.laion2b_ft_in1k
# Train the donors and cache the backbone features first. Uses the GPU in CUDA_VISIBLE_DEVICES.
set -euo pipefail
cd "$(dirname "$0")/../.."

DONOR_KIND=$1; COND=$2; SEED=$3; CNN=${4:-resnet50_clip.openai}
ONE_HOT=data/checkpoints/abstract/hs128_lstm_lr3e-3_wd1e-4_nomturk_rep${SEED}

matched() {  # <run dir> <accuracy> [any_epoch]: the checkpoint the analysis code selects
  python -c "import sys; from pathlib import Path; from analysis.factory import find_best_checkpoint as f; \
print(f(Path(sys.argv[1]), 'final_RNN_model', int(sys.argv[2]), acc_match=float(sys.argv[3]), any_epoch=sys.argv[4] == '1'))" \
    "$1" "$SEED" "$2" "${3:-0}"
}

case $DONOR_KIND in
  final)    TAG=abfinal;  DONOR=$ONE_HOT/final_RNN_model_rep${SEED}.pth ;;
  m80)      TAG=abm80;    DONOR=$(matched data/checkpoints/abstract_until080/hs128_lstm_lr3e-3_wd1e-4_nomturk_rep${SEED} 0.80 1) ;;
  match075) TAG=abmatch;  DONOR=$(matched "$ONE_HOT" 0.75) ;;
  dimmatch) TAG=dimmatch; DONOR=data/checkpoints/dim_match/hs128_lstm_lr3e-4_wd1e-4_dim2048_ns0.35_k0.5_nomturk_rep${SEED}/final_RNN_model_rep${SEED}.pth ;;
  *) echo "donor must be final, m80, match075 or dimmatch" >&2; exit 1 ;;
esac

PRE=data/checkpoints/abs_to_nat/cnntype_${CNN}_hs128_lstm_lr3e-4_wd1e-4_frozen
case $COND in
  theta)     OUT=${PRE}_${TAG}_nomturk_ls0.2_rep${SEED};           EXTRA="--seq_model_path $DONOR" ;;
  wout)      OUT=${PRE}_${TAG}_nomturk_ls0.2_frzfcout_rep${SEED};  EXTRA="--seq_model_path $DONOR --freeze_fc_out" ;;
  win)       [ "$DONOR_KIND" = dimmatch ] && [ "$CNN" = resnet50_clip.openai ] \
               || { echo "win needs the dimmatch donor and resnet50_clip.openai" >&2; exit 1; }
             OUT=${PRE}_${TAG}_nomturk_ls0.2_frzinpath_rep${SEED}; EXTRA="--seq_model_path $DONOR --freeze_input_pathway" ;;
  untrained) OUT=${PRE}_dimmatch_nomturk_ls0.2_randseq_rep${SEED}; EXTRA="" ;;
  *) echo "condition must be theta, wout, win or untrained" >&2; exit 1 ;;
esac
if [ "$COND" != untrained ] && [ ! -f "$DONOR" ]; then echo "missing donor checkpoint $DONOR" >&2; exit 1; fi
mkdir -p "$OUT"

# shellcheck disable=SC2086
python train_natural_frozen.py \
  --batch_size 128 --train_dataset_size 256000 --val_dataset_size 64000 --num_epochs 20 \
  --lr 3e-4 --weight_decay 1e-4 --dropout 0.3 --grad_norm 2.0 --warmup_epochs 0 \
  --cnn_type "$CNN" --hidden_size 128 --seq_model_type lstm --num_heads 8 \
  --freeze_seq_model --label_smoothing 0.2 --num_workers 6 --pin_mem \
  --rep_index "$SEED" --results_dir "$OUT" $EXTRA \
  2>&1 | tee "$OUT/out.txt"
