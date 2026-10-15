#!/usr/bin/env bash
# DVCE settings test on 6 diatoms (evaluation/vce_pairs_tune6.json); one setting after another.
cd "$(dirname "$0")/.."
export PAIRS=evaluation/vce_pairs_tune6.json
NAME=tune_T1_skip130  EXTRA="--skip_timesteps 130"        bash diatoms/run_dvce_experiment.sh
NAME=tune_T2_skip150  EXTRA="--skip_timesteps 150"        bash diatoms/run_dvce_experiment.sh
NAME=tune_T3_clf0.05  EXTRA="--classifier_lambda 0.05"    bash diatoms/run_dvce_experiment.sh
NAME=tune_T4_lp0.3    EXTRA="--lp_custom_value 0.3"       bash diatoms/run_dvce_experiment.sh
echo TUNING DONE
