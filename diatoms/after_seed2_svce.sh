#!/usr/bin/env bash
# After the full-run seed-2 DVCE run: SVCE baseline on the 50-diatom pairs (both GPUs), then
# comparisons (main vs SVCE, clean evaluator, traits, background test) and the seed-2 statistics.
#   setsid nohup bash diatoms/after_seed2_svce.sh > logs/after_seed2_svce.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/python_38_dvces_cu111/bin/python
say() { echo "[$(date '+%F %T')] $*"; }
R=DiatlasVCEs/clf_r50madry_random_eps1.0__diff_ft_v2_nativesize
D=L2_0_l1_0_l1.0_0.15_classifier_3_-1_-1_0.1_reg_lpips_0_example_200_steps_skip_100_start_p_sample_deg_0_s
V=L1.5_0_l1_0_l1.0_0.15_classifier_3_-1_-1_0.1_reg_lpips_0_example_200_steps_skip_100_start_p_sample_deg_0_s_1
E=DiatlasModels/r50imagenet_page_eps0/best.pt

say "waiting for the full seed-2 run"
until grep -q "all done" logs/dvce_full_seed2.out 2>/dev/null; do sleep 60; done

say "SVCE baseline, 50 diatoms, both GPUs"
NAME=svce_r50eps1 EXTRA="--config svce.yml" bash diatoms/run_dvce_experiment.sh > logs/dvce_svce.out 2>&1

say "comparisons"
MAIN=$R/${D}_1_wid_0_2_dvces/50.pth,$R/${D}_1_wid_1_2_dvces/50.pth
SVCE=$R/${V}_wid_0_2_svces/50.pth,$R/${V}_wid_1_2_svces/50.pth
mkdir -p results/dvce/svce_comparison
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 $PY diatoms/compare_dvce_runs.py --run main:$MAIN --run svce:$SVCE \
    --eval_ckpt $E DiatlasModels/resnet18_native_eps0.5/best.pt --eval_name clean_r50_page robust_r18 \
    --out_dir results/dvce/svce_comparison 2>/dev/null
CUDA_VISIBLE_DEVICES= $PY diatoms/trait_check.py --run main50:$MAIN --run svce:$SVCE --out_dir results/traits_svce 2>/dev/null >/dev/null

say "seed-2 statistics for the full run"
F1=$R/${D}_1_wid_0_2_dvces/302.pth,$R/${D}_1_wid_1_2_dvces/300.pth
F2=$R/${D}_2_wid_0_2_dvces/302.pth,$R/${D}_2_wid_1_2_dvces/300.pth
mkdir -p results/dvce/full_seeds
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 $PY diatoms/compare_dvce_runs.py --run seed1:$F1 --run seed2:$F2 \
    --eval_ckpt $E --eval_name clean_r50_page --fig_sources --out_dir results/dvce/full_seeds 2>/dev/null
CUDA_VISIBLE_DEVICES= $PY diatoms/trait_check.py --run seed1:$F1 --run seed2:$F2 --out_dir results/traits_seeds 2>/dev/null >/dev/null
say "all done"
