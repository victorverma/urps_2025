#!/bin/bash

conda env create -p autogluon_env/ -f autogluon_env.yml
conda activate autogluon_env/
pip install catboost==1.2.7 --only-binary :all:
uv pip install autogluon.timeseries
conda deactivate
