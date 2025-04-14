#!/bin/sh

conda run -p ../../autogluon_env/ --live-stream python multi_model.py \
    --num_runs=1000 \
    --fve=0.5 \
    --train_size=10000 \
    --prediction_length=100 \
    --eval_metric="RMSE" \
    --ci_level=0.95 \
    --time_limit=600 \
    --max_epochs=1 \
    --verbosity=4 \
    --max_workers=$SLURM_NTASKS_PER_NODE
