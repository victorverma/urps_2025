#!/bin/sh

conda env create -p autogluon_env/ -f autogluon_env.yml
conda run -p autogluon_env/ pip install catboost==1.2.7 --only-binary :all:
conda run -p autogluon_env/ uv pip install autogluon.timeseries
