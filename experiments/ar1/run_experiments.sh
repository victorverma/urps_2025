#!/bin/sh

conda run -p ../../autogluon_env/ --live-stream python run_experiment_4.py \
    --num_runs=2 \
    --fve=0.9 \
    --train_size=1000 \
    --prediction_length=100 \
    --eval_metric="RMSE" \
    --ci_level=0.95 \
    --time_limit=600 \
    --max_epochs=5 \
    --verbosity=4 \
    --max_workers=10
