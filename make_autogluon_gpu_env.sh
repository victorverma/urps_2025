#!/bin/sh

conda env create -p autogluon_gpu_env/ -f autogluon_gpu_env.yml
conda run -p autogluon_gpu_env/ uv pip install autogluon.timeseries
