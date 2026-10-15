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
set -euo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/python_38_dvces_cu111/bin/python
NAME=${NAME:?set NAME}
PAIRS=${PAIRS:-evaluation/vce_pairs_paper50.json}
CLASSIFIER=${CLASSIFIER:-DiatlasModels/r50madry_random_eps1.0/best.pt}
DIFFUSION=${DIFFUSION:-checkpoints/diatlas/ft_v2_nativesize/ema_0.9995_030000.pt}
EXTRA=${EXTRA:-}
N=$($PY -c "import json; print(len(json.load(open('$PAIRS'))['entries']))")
PER_GPU=$(( (N + 1) / 2 * 2 ))   # counterfactuals per GPU (2 per diatom)
OUT=results/dvce/$NAME
mkdir -p "$OUT" logs
echo "$(date) | $NAME: $N diatoms, $((2 * N)) counterfactuals, classifier $CLASSIFIER, diffusion $DIFFUSION $EXTRA" | tee "$OUT/run_info.txt"

for W in 0 1; do
    CUDA_VISIBLE_DEVICES=$W $PY diatlas_VCEs.py --gpu 0 --vce_list "$PAIRS" --num_imgs $PER_GPU --batch_size 2 \
        --diatom_classifier "$CLASSIFIER" --diffusion_ckpt "$DIFFUSION" --denoise_dist_input \
        --world_size 2 --world_id $W $EXTRA > "logs/dvce_${NAME}_gpu$W.out" 2>&1 &
done
wait
echo "$(date) | DVCE done" | tee -a "$OUT/run_info.txt"

CLF_DIR=DiatlasVCEs/clf_$(basename "$(dirname "$CLASSIFIER")")__diff_$(basename "$(dirname "$DIFFUSION")")
PTHS=$(ls -t $CLF_DIR/*wid_0_2*/*.pth $CLF_DIR/*wid_1_2*/*.pth 2>/dev/null | head -2)
echo "result files: $PTHS" | tee -a "$OUT/run_info.txt"
CUDA_VISIBLE_DEVICES=0 $PY diatoms/dvce_postprocess.py --pth $PTHS --ckpt "$CLASSIFIER" --out_dir "$OUT" 2>/dev/null
echo "$(date) | all done -> $OUT" | tee -a "$OUT/run_info.txt"
