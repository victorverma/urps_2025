import argparse
import numpy as np
import os
import re
import pandas as pd
import shutil
import time
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from concurrent.futures import ProcessPoolExecutor
from plotnine import *
from statsmodels.tsa.arima_process import ArmaProcess
from typing import Any, Dict, Type
import random
import torch
import pickle
import gluonts.torch.model.patch_tst
import torch.nn as nn
from ptflops import get_model_complexity_info

class WrappedForecastModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape

        # Dummy values for required model inputs
        return self.model(
            past_target=x.squeeze(-1),
            past_observed_values=torch.ones_like(x.squeeze(-1)),  # Typically a mask of same shape
        )

def cal_flops(hyperparameters: Dict[str | Type, Any], prediction_length: int) -> list:
    hyperparameters = hyperparameters["PatchTST"]
    model = gluonts.torch.model.patch_tst.PatchTSTModel(
        context_length=hyperparameters["context_length"], 
        prediction_length=prediction_length, 
        patch_len=1, 
        stride=hyperparameters["stride"],
        d_model=hyperparameters["d_model"], 
        nhead=hyperparameters["nhead"], 
        num_encoder_layers=hyperparameters["num_encoder_layers"], 
        scaling="mean",
        activation="relu", 
        dim_feedforward=128, 
        dropout=0.1, 
        norm_first=False, 
        padding_patch="end", 
        num_feat_dynamic_real=0
    )

    wrapped_model = WrappedForecastModel(model)

    with torch.no_grad():
        macs, params = get_model_complexity_info(wrapped_model, (hyperparameters["context_length"], 1), as_strings=True)
        print(f"MACs: {macs}, Parameters: {params}")

    mac_match = re.search(r"(\d+(\.\d+)?)", macs)
    flops = 2*float(mac_match.group(1))
    print(f"FLOPs: {2*flops}") # FLOPs=2×MACs

    param_match = re.search(r"(\d+(\.\d+)?)", params)
    num_params = float(param_match.group(1))
    return [flops, num_params]

def rmse(y_true, y_pred):
    """Compute RMSE (Root Mean Squared Error)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def help_check_prediction_intervals(test_data: TimeSeriesDataFrame,
                                     prediction_length: int, predictions: TimeSeriesDataFrame) -> pd.Series:
    actuals = test_data["target"]
    are_lower_bounds_right = actuals >= predictions.iloc[:, -2]
    are_upper_bounds_right = actuals <= predictions.iloc[:, -1]
    checks = are_lower_bounds_right & are_upper_bounds_right
    checks.index = [str(i) for i in range(1, prediction_length + 1)]
    return checks

def check_prediction_intervals(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        prediction_length: int,
        eval_metric: str,
        ci_level: float,
        time_limit: int,
        verbosity: int,
        hyperparameters: Dict[str | Type, Any]
    ) -> pd.DataFrame:

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        eval_metric=eval_metric,
        verbosity=verbosity,
        quantile_levels=[(1 - ci_level) / 2, (1 + ci_level) / 2]
    )
    
    predictor.fit(train_data=train_df, time_limit=time_limit, hyperparameters=hyperparameters, enable_ensemble=False)

    results_list = []
    for model_name, _ in hyperparameters.items():            
        predictions = predictor.predict(train_df, model=model_name)
     
        coverage_flags = help_check_prediction_intervals(test_df, prediction_length, predictions)
        flops, num_params = cal_flops(model_name, hyperparameters[model_name], prediction_length)
        
        df = pd.DataFrame(
            {"model_name": model_name,
            "h": range(1, prediction_length+1),
            "observed":test_df["target"].iloc[-prediction_length:],
            "prediction": predictions["mean"].values, 
            "lower_bound": lower_bounds.values, 
            "upper_bound": upper_bounds.values,
            "coverage_flags": coverage_flags.values, 
            "squared_error": (test_df["target"].iloc[-prediction_length:] - predictions.iloc[:, 0]) ** 2,
            "pred_length": upper_bounds.values - lower_bounds.values,
            "flops": flops,
            "num_params": num_params,
            # variance of sample: sd
            "sd": np.std(train_df["target"]),
            "runtime": predictor.fit_summary()["model_fit_times"][model_name]
        }        
        )

        results_list.append(df)

    return pd.concat(results_list)


def do_1_run(
        run_idx: int,
        train_size: int,
        prediction_length: int,
        eval_metric: str,
        ci_level: float,
        time_limit: int,
        verbosity: int,
        train_full_df: pd.DataFrame,
        test_full_df: pd.DataFrame,
        hyparam_data: pd.DataFrame 
    ) -> pd.DataFrame:

    # extract the same dataset for each model training & testing
    train_start_idx = 0 
    train_end_idx = train_size
    train_df = train_full_df.iloc[train_start_idx:train_end_idx, ]
    
    test_start_idx = 0 
    test_end_idx = prediction_length
    test_df = test_full_df.iloc[test_start_idx:test_end_idx, ]

    # extract the random generated hyperparameters of PatchTST for each model
    patch_len = 1  # Static value for AR(1)
    stride = hyparam_data["stride"][run_idx]
    nhead = hyparam_data["nhead"][run_idx]
    d_model = hyparam_data["d_model"][run_idx]
    context_length = hyparam_data["context_length"][run_idx]
    num_encoder_layers = hyparam_data["num_encoder_layers"][run_idx]

    # Ensure d_model is divisible by nhead
    assert d_model % nhead == 0

    hyperparameters = {
        # "AutoARIMA": { }, 
        "PatchTST": {
            'max_epochs': max_epochs,
            'patch_len': patch_len,
            'stride': stride,
            'nhead': nhead,
            'd_model': d_model,
            'context_length': context_length,
            'num_encoder_layers': num_encoder_layers,
            }
    # "TemporalFusionTransformer": {"max_epochs": max_epochs}
    }

    checks = check_prediction_intervals(train_df, test_df, prediction_length, eval_metric, ci_level, time_limit, verbosity, hyperparameters)
    checks.insert(0, "run_idx", run_idx)
    return checks



def evaluate_hyperparam(
        train_size: int,
        prediction_length: int,
        eval_metric: str,
        ci_level: float,
        time_limit: int,
        verbosity: int,
        train_full_df: pd.DataFrame,
        test_full_df: pd.DataFrame,
        hyperparameters: dict,
        hyper_idx: int,
        dir_name: str
    ) -> pd.DataFrame:

    # extract the same dataset for each model training & testing
    # step 1 : training over the first dataset
    train_start_idx = 0 
    train_end_idx = train_size
    train_df = train_full_df.iloc[train_start_idx:train_end_idx, :]
    test_df = test_full_df.iloc[0 : prediction_length, :]

    train_df = pd.concat([train_df, test_df])

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        eval_metric=eval_metric,
        verbosity=verbosity,
        quantile_levels=[(1 - ci_level) / 2, (1 + ci_level) / 2]
    )
    
    predictor.fit(train_data=train_df, time_limit=time_limit, hyperparameters=hyperparameters, enable_ensemble=False)

    # step 2: testing over the rest dataset
    num_test_dataset = 100 #len(test_full_df["dataset_idx"].unique())
    results_list = []
    
    for i in range(num_test_dataset):
        observed_start_idx = i * train_size
        observed_end_idx = observed_start_idx + train_size
        observed_df = train_full_df.iloc[observed_start_idx:observed_end_idx, :]

        test_df = test_full_df.iloc[i*prediction_length : (i+1)*prediction_length, :]

        predictions = predictor.predict(observed_df)
        lower_bounds, upper_bounds = predictions.iloc[:, -2], predictions.iloc[:, -1]

        coverage_flags = help_check_prediction_intervals(test_df, prediction_length, predictions)
  
        flops, num_params = cal_flops(hyperparameters, prediction_length)
        runtime = next(iter(predictor.fit_summary()["model_fit_times"])) # extract runtime from dict of dict
        df = pd.DataFrame(
            {"hyper_idx": hyper_idx,
            "h": range(1, prediction_length+1),
            "observed":test_df["target"].values,
            "prediction": predictions["mean"].values, 
            "lower_bound": lower_bounds.values, 
            "upper_bound": upper_bounds.values,
            "coverage_flags": coverage_flags.values, 
            "squared_error": (test_df["target"].values - predictions.iloc[:, 0].values) ** 2,
            "pred_length": upper_bounds.values - lower_bounds.values,
            "flops": flops,
            "num_params": num_params,
            "sd": np.std(train_df["target"].values), # variance of sample: sd
            "runtime": runtime
        }        
        )

        results_list.append(df)

    results = pd.concat(results_list)
    results.to_parquet(f"{dir_name}/results_hyper_idx_{hyper_idx}.parquet")

    return results
    

def get_hyperparameter(df_hyperparameter: pd.DataFrame, max_epochs: int) -> dict:
    patch_len = 1  # Static value for AR(1)
    stride = df_hyperparameter["stride"]
    nhead = df_hyperparameter["nhead"]
    d_model = df_hyperparameter["d_model"]
    context_length = df_hyperparameter["context_length"]
    num_encoder_layers = df_hyperparameter["num_encoder_layers"]

    hyperparameters = {
        "PatchTST": {
            'max_epochs': max_epochs,
            'patch_len': patch_len,
            'stride': stride,
            'nhead': nhead,
            'd_model': d_model,
            'context_length': context_length,
            'num_encoder_layers': num_encoder_layers,
            }
    }
    return hyperparameters


def run_experiment(
        num_runs: int,
        fve: float,
        train_size: int,
        prediction_length: int,
        eval_metric: str,
        ci_level: float,
        time_limit: int,
        verbosity: int,
        max_workers: int#,
        #max_epochs: int
    ) -> pd.DataFrame:

    # load in full train and test data
    train_full_df = f"./input_data/train_1000_{fve}_{train_size}_{prediction_length}.parquet"
    test_full_df = f"./input_data/test_1000_{fve}_{train_size}_{prediction_length}.parquet"
    patch_hyperparam_df = pd.read_parquet(f"./input_data/PatchTST_hyperparams_1000.parquet")
    num_hyper = len(patch_hyperparam_df.index)
    train_full_df = TimeSeriesDataFrame.from_data_frame(train_full_df)
    test_full_df = TimeSeriesDataFrame.from_data_frame(test_full_df)

    dir_name = make_dir_name(num_runs, fve, train_size, prediction_length, eval_metric, ci_level, time_limit, max_epochs)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    if max_workers == 1:
        results = []
        for hyper_idx in range(num_hyper):
            hyperparameters = get_hyperparameter(patch_hyperparam_df.iloc[hyper_idx, :], max_epochs)
            result = evaluate_hyperparam(train_size, prediction_length, eval_metric, ci_level, time_limit,
            verbosity, train_full_df, test_full_df, hyperparameters, hyper_idx, dir_name)
            results.append(result)

    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    do_1_run,
                    range(num_runs),
                    [train_size] * num_runs,
                    [prediction_length] * num_runs,
                    [eval_metric] * num_runs,
                    [ci_level] * num_runs,
                    [time_limit] * num_runs,
                    [verbosity] * num_runs,
                    [train_full_df] * num_runs,
                    [test_full_df] * num_runs,
                    [patch_hyperparam_df] * num_runs
                )
            )
    results = pd.concat(results, ignore_index=True)

    return results

def make_dir_name(
        num_runs: int,
        fve: float,
        train_size: int,
        prediction_length: int,
        eval_metric: str,
        ci_level: float,
        time_limit: int,
        max_epochs: int
    ) -> str:
    dir_name = f"multi_model_{num_runs}_{fve}_{train_size}_{prediction_length}_{eval_metric}_{ci_level}_{time_limit}_{max_epochs}"
    return dir_name

if __name__ == "__main__":
    print("Running AR(1) Experiment...")
    experiment_start_time = time.time()

    ################################################################################
    # Parse the command-line arguments
    ################################################################################

    parser = argparse.ArgumentParser(description="Run an AR(1) experiment")
    parser.add_argument("--num_runs", type=int, required=True, help="Number of runs to execute")
    parser.add_argument("--fve", type=float, required=True, help="Fraction of variance explained of the AR(1) model, e.g., 0.5")
    parser.add_argument("--train_size", type=int, required=True, help="Size of a training set")
    parser.add_argument(
        "--prediction_length", type=int, required=True, help="Number of time steps after a training set to compute predictions at"
    )
    parser.add_argument("--eval_metric", type=str, required=True, help="Metric to use for hyperparameter tuning on a validation set")
    parser.add_argument("--ci_level", type=float, required=True, help="Level of the prediction intervals, e.g., 0.95")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum number of seconds for fitting all of the models on one run")
    parser.add_argument("--max_epochs", type=int, required=True, help="Maximum number of epochs for fitting a transformer-based model")
    parser.add_argument(
        "--verbosity", default=0, type=int, choices=range(5), help="Level of detail of printed information about model fitting"
    )
    parser.add_argument("--max_workers", default=1, type=int, help="Maximum number of worker processes to use")

    cmd_args = parser.parse_args()
    num_runs = cmd_args.num_runs
    fve = cmd_args.fve
    train_size = cmd_args.train_size
    prediction_length = cmd_args.prediction_length
    eval_metric = cmd_args.eval_metric
    ci_level = cmd_args.ci_level
    time_limit = cmd_args.time_limit
    max_epochs = cmd_args.max_epochs
    verbosity = cmd_args.verbosity
    max_workers = cmd_args.max_workers

    ################################################################################
    # Run the experiment
    ################################################################################

    results = run_experiment(
        num_runs, fve, train_size, prediction_length, eval_metric, ci_level, time_limit, verbosity, max_workers
    )

    ################################################################################
    # Save the results
    ################################################################################

    dir_name = make_dir_name(num_runs, fve, train_size, prediction_length, eval_metric, ci_level, time_limit, max_epochs)

    results.to_parquet(f"{dir_name}/results.parquet")

    experiment_elapsed_time = time.time() - experiment_start_time
    print(f"Done ({int(experiment_elapsed_time)}s)")
