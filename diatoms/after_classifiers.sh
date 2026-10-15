#!/usr/bin/env bash
# Runs when the random-split classifiers finish (2026-10-13): classifier test
# evaluation on GPU 0, native-size reconstruction check of ft_v1 on GPU 1.
set -u
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/python_38_dvces_cu111/bin/python
while ! { grep -q "training finished" DiatlasModels/r50madry_random_eps0.5/log.txt && grep -q "training finished" DiatlasModels/r50madry_random_eps1.0/log.txt; }; do
    ps -eo args | grep -q "[r]50madry_random" || break
    sleep 60
done
mkdir -p results/classifier results/diffusion
CUDA_VISIBLE_DEVICES=0 $PY diatoms/eval_classifier.py --split test --split_mode random \
    --ckpt independent --ckpt DiatlasModels/r50madry_random_eps0.5/best.pt --ckpt DiatlasModels/r50madry_random_eps1.0/best.pt \
    --eps 0.25 0.5 1.0 2.0 2>/dev/null | tee results/classifier/test_r50madry_random_curve.txt &
(
for ck in imagenet:checkpoints/256x256_diffusion_uncond.pt ft_v1:checkpoints/diatlas/ft_v1/ema_0.9995_030000.pt; do
    CUDA_VISIBLE_DEVICES=1 $PY diatoms/sample_diffusion.py --ckpt ${ck#*:} --mode recon --n 32 --batch_size 8 \
        --data_root data/diatlas_nativesize_crop224_canvas256 --out results/diffusion/native_recon_${ck%%:*}.png 2>/dev/null | grep saved
done
CUDA_VISIBLE_DEVICES= $PY diatoms/recon_metrics.py \
    --grid results/diffusion/native_recon_imagenet.png --name imagenet \
    --grid results/diffusion/native_recon_ft_v1.png --name ft_v1 \
    --masks_out results/diffusion/native_recon_masks.png 2>/dev/null | tee results/diffusion/native_recon_metrics.txt
) &
wait
echo ALL DONE
