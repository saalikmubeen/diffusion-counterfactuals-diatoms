#!/usr/bin/env bash
# Full DVCE experiment in one command: DVCE on both GPUs (half of the pairs each),
# then one figure per source diatom and a metrics table + summary.
#
#   NAME=main_r50eps1_ftv2 bash diatoms/run_dvce_experiment.sh
#
# Settings (environment variables, defaults = main experiment):
#   NAME        output name -> results/dvce/$NAME/
#   PAIRS       pair list (JSON)               default evaluation/vce_pairs_paper50.json
#   CLASSIFIER  classifier to explain          default DiatlasModels/r50madry_random_eps1.0/best.pt
#   DIFFUSION   diffusion model                default checkpoints/diatlas/ft_v2_nativesize/ema_0.9995_030000.pt
#   EXTRA       extra diatlas_VCEs.py arguments (e.g. "--diatom_classifier_2 independent --deg_cone_projection 30")
#   GPU         if set, run the halves one after the other on this GPU only
#               (default: half 0 on GPU 0 and half 1 on GPU 1 in parallel)
#   SHARDS      which halves to (re)run, default "0 1" (e.g. SHARDS=1 GPU=0 to redo half 1 on GPU 0)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/python_38_dvces_cu111/bin/python
NAME=${NAME:?set NAME}
PAIRS=${PAIRS:-evaluation/vce_pairs_paper50.json}
CLASSIFIER=${CLASSIFIER:-DiatlasModels/r50madry_random_eps1.0/best.pt}
DIFFUSION=${DIFFUSION:-checkpoints/diatlas/ft_v2_nativesize/ema_0.9995_030000.pt}
EXTRA=${EXTRA:-}
GPU=${GPU:-}
SHARDS=${SHARDS:-0 1}
N=$($PY -c "import json; print(len(json.load(open('$PAIRS'))['entries']))")
PER_GPU=$(( (N + 1) / 2 * 2 ))   # counterfactuals per GPU (2 per diatom)
OUT=results/dvce/$NAME
mkdir -p "$OUT" logs
echo "$(date) | $NAME: $N diatoms, $((2 * N)) counterfactuals, classifier $CLASSIFIER, diffusion $DIFFUSION $EXTRA" | tee "$OUT/run_info.txt"

run_shard() {
    CUDA_VISIBLE_DEVICES=$2 $PY diatlas_VCEs.py --gpu 0 --vce_list "$PAIRS" --num_imgs $PER_GPU --batch_size 2 \
        --diatom_classifier "$CLASSIFIER" --diffusion_ckpt "$DIFFUSION" --denoise_dist_input \
        --world_size 2 --world_id $1 $EXTRA > "logs/dvce_${NAME}_half$1.out" 2>&1
}
START=$(mktemp)                     # marker: only result files written after this count
FAIL=0
PIDS=()
for W in $SHARDS; do
    if [ -n "$GPU" ]; then run_shard $W $GPU || FAIL=1; else run_shard $W $W & PIDS+=($!); fi
done
for P in "${PIDS[@]}"; do wait "$P" || FAIL=1; done
if [ "$FAIL" != 0 ]; then
    echo "$(date) | DVCE FAILED (see logs/dvce_${NAME}_half*.out); not post-processing" | tee -a "$OUT/run_info.txt"
    exit 1
fi
echo "$(date) | DVCE done" | tee -a "$OUT/run_info.txt"

CLF_NAME=$(basename "$(dirname "$CLASSIFIER")")
[ "$CLF_NAME" = "." ] && CLF_NAME=$CLASSIFIER       # e.g. CLASSIFIER=independent (same rule as diatlas_VCEs.py)
CLF_DIR=DiatlasVCEs/clf_${CLF_NAME}__diff_$(basename "$(dirname "$DIFFUSION")")
PTHS=$(find $CLF_DIR -path '*wid_[01]_2*' -name '*.pth' -newer "$START" | sort)
if [ "$(echo "$PTHS" | grep -c .)" != 2 ]; then
    echo "$(date) | expected 2 new result files, found: $PTHS" | tee -a "$OUT/run_info.txt"
    exit 1
fi
echo "result files: $PTHS" | tee -a "$OUT/run_info.txt"
CUDA_VISIBLE_DEVICES=0 $PY diatoms/dvce_postprocess.py --pth $PTHS --ckpt "$CLASSIFIER" --out_dir "$OUT" 2>/dev/null
echo "$(date) | all done -> $OUT" | tee -a "$OUT/run_info.txt"
