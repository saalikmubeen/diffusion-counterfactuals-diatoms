#!/usr/bin/env bash
# Overnight chain (2026-10-21): page-split classifier experiment + clean preprocessing test.
# Waits for the jobs already running, then:
#   1. trains the shrink-to-fit preprocessing arm (GPU 1, after the stretched arm finishes)
#   2. evaluates the page-split classifiers on the page-split test set (GPU 0)
#   3. runs DVCE with the page-split robust ResNet-50 on the same 50 diatoms (GPU 0)
#   4. compares explanations: main vs seed 2 (noise floor) vs page-split classifier, plus
#      transfer to the clean evaluator and the trait check
# Everything is logged to logs/overnight_pagesplit.log; key numbers to results/overnight_summary.txt.
#
#   setsid nohup bash diatoms/overnight_pagesplit.sh > logs/overnight_pagesplit.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/python_38_dvces_cu111/bin/python
say() { echo "[$(date '+%F %T')] $*"; }
finished() { grep -q "training finished" "DiatlasModels/$1/log.txt" 2>/dev/null; }

S=L2_0_l1_0_l1.0_0.15_classifier_3_-1_-1_0.1_reg_lpips_0_example_200_steps_skip_100_start_p_sample_deg_0_s
pths() { echo "$1/${S}_$2_wid_0_2_dvces/50.pth,$1/${S}_$2_wid_1_2_dvces/50.pth"; }   # dir-prefix, seed
RAND=DiatlasVCEs/clf_r50madry_random_eps1.0__diff_ft_v2_nativesize/
PAGE=DiatlasVCEs/clf_r50madry_page_eps1.0__diff_ft_v2_nativesize/

# 1. shrink-to-fit arm on GPU 1 once the stretched arm is done
say "waiting for the stretched arm"
until finished r50imagenet_page_eps0_stretched; do sleep 60; done
say "training shrink-to-fit arm (GPU 1)"
CUDA_VISIBLE_DEVICES=1 $PY diatoms/train_classifier.py --run r50imagenet_page_eps0_fit --preproc nativefit \
    --split_mode page --init imagenet_r50 --eps 0 --augment strong --epochs 15 > logs/cls_r50imagenet_page_eps0_fit.out 2>&1 &
FIT_PID=$!

# 2. page-split robust classifier: wait, evaluate
say "waiting for the page-split robust ResNet-50"
until finished r50madry_page_eps1.0; do sleep 60; done
say "evaluating page-split classifiers on the page-split test set (GPU 0)"
CUDA_VISIBLE_DEVICES=0 $PY diatoms/eval_classifier.py --split test --split_mode page \
    --ckpt DiatlasModels/r50madry_page_eps1.0/best.pt --ckpt DiatlasModels/r50imagenet_page_eps0/best.pt \
    --eps 0.5 1.0 > results/classifier/test_page_models.txt 2> logs/eval_page_models.err

# 3. DVCE with the page-split classifier, same 50 pairs, GPU 0 (GPU 1 still has the seed-2 run)
say "DVCE with the page-split classifier (GPU 0, ~2 h)"
GPU=0 NAME=pagesplit_r50eps1_ftv2 CLASSIFIER=DiatlasModels/r50madry_page_eps1.0/best.pt \
    bash diatoms/run_dvce_experiment.sh > logs/dvce_pagesplit.out 2>&1
say "page-split DVCE done"

# 4. comparisons (CPU, after the seed-2 run is complete)
say "waiting for the seed-2 run"
until grep -q "all done" logs/dvce_seed2.out 2>/dev/null; do sleep 60; done
wait $FIT_PID
say "comparisons"
mkdir -p results/dvce/pagesplit_comparison results/traits_pagesplit
EV="DiatlasModels/r50imagenet_page_eps0/best.pt DiatlasModels/resnet18_native_eps0.5/best.pt DiatlasModels/r50madry_random_eps0.5/best.pt DiatlasModels/r50madry_page_eps1.0/best.pt"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 $PY diatoms/compare_dvce_runs.py \
    --run main:$(pths ${RAND%/} 1) --run seed2:$(pths ${RAND%/} 2) --run pagesplit_clf:$(pths ${PAGE%/} 1) \
    --eval_ckpt $EV --eval_name clean_r50_page robust_r18 robust_r50_rand_e05 robust_r50_page_e1 \
    --out_dir results/dvce/pagesplit_comparison 2>/dev/null
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 $PY diatoms/compare_explanations.py \
    --ref main:$(pths ${RAND%/} 1) --run seed2:$(pths ${RAND%/} 2) --run pagesplit_clf:$(pths ${PAGE%/} 1) \
    --run imagenet_prior:$(pths DiatlasVCEs/clf_r50madry_random_eps1.0__diff_checkpoints 1) \
    --run nonrobust_clf:$(pths DiatlasVCEs/clf_independent__diff_ft_v2_nativesize 1) \
    --out results/dvce/pagesplit_comparison/explanation_agreement.txt 2>/dev/null
CUDA_VISIBLE_DEVICES= $PY diatoms/trait_check.py --run main50:$(pths ${RAND%/} 1) --run seed2:$(pths ${RAND%/} 2) \
    --run pagesplit_clf:$(pths ${PAGE%/} 1) --out_dir results/traits_pagesplit 2>/dev/null > /dev/null

# summary
{
  echo "OVERNIGHT SUMMARY ($(date '+%F %T'))"; echo
  echo "== Preprocessing, clean test (ImageNet ResNet-50, page split, 15 epochs): best val accuracy"
  for r in r50imagenet_page_eps0 r50imagenet_page_eps0_stretched r50imagenet_page_eps0_fit; do
    echo "  $r: $(grep -o 'best val acc [0-9.]*' DiatlasModels/$r/log.txt | tail -1)"; done
  echo; echo "== Page-split classifiers, page-split test set"; cat results/classifier/test_page_models.txt
  echo; echo "== DVCE with the page-split classifier (own summary)"; cat results/dvce/pagesplit_r50eps1_ftv2/summary.txt
  echo; echo "== Same 50 pairs: main vs seed 2 vs page-split classifier"; cat results/dvce/pagesplit_comparison/comparison.txt
  echo; echo "== Do explanations change? (reference: main run)"; cat results/dvce/pagesplit_comparison/explanation_agreement.txt
  echo; echo "== Trait check"; cat results/traits_pagesplit/summary.txt
} > results/overnight_summary.txt 2>&1
say "all done -> results/overnight_summary.txt"
