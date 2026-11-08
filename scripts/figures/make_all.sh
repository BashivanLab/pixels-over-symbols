#!/usr/bin/env bash
# Draw every figure of the paper from the cached results in data/evaluations, on the CPU.
# Figures go to figures/, one log per job to figures/logs/. JOBS jobs run at once (default 4).
#
#     bash scripts/figures/make_all.sh
set -u
cd "$(dirname "$0")/../.."
# the evaluation pipeline's RSA results depend on the hash seed (see load() in behavior.py)
export PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
JOBS="${JOBS:-4}"
LOGS=figures/logs
mkdir -p "$LOGS"
rm -f "$LOGS/failed"

run() {  # run <name> <command...> in the background once fewer than $JOBS jobs are running
  while [ "$(jobs -pr | wc -l)" -ge "$JOBS" ]; do sleep 1; done
  local name=$1
  shift
  echo "start $name"
  ( "$@" > "$LOGS/$name.log" 2>&1 || echo "$name" >> "$LOGS/failed" ) &
}

for set in seq_type ab_match_levels vision_type seq_size encoder_size noise_dose metric_alignment \
           transplant_final transplant_acc_matched transplant_dm; do
  run "$set" python scripts/figures/behavior.py "$set"
done
run dsa_transplant python scripts/figures/dsa_transplant.py
run dsa_heatmap python scripts/figures/dsa_heatmap.py
wait

if [ -s "$LOGS/failed" ]; then
  echo "failed: $(tr '\n' ' ' < "$LOGS/failed")(logs in $LOGS/)"
  exit 1
fi
echo "done; figures in figures/, logs in $LOGS/"
