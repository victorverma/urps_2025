#!/bin/sh

conda run -p ../../autogluon_env/ --live-stream python run_experiment_jake.py \
    --num_runs=10 \
    --fve=0.9 \
    --train_size=1000 \
    --prediction_length=100 \
    --eval_metric="RMSE" \
    --ci_level=0.95 \
    --time_limit=60 \
