import argparse
import numpy as np
import os
import pandas as pd
import shutil
import time
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from plotnine import *
from statsmodels.tsa.arima_process import ArmaProcess
from typing import Any, Dict, Type

def simulate_ar1(phi: float, sigma: float, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if np.abs(phi) >= 1:
        raise ValueError("The absolute value of phi must be less than one.")
    arma_process = ArmaProcess(ar=np.array([1, -phi]), ma = np.array([1]))
    targets = arma_process.generate_sample(n, sigma, distrvs=lambda size: rng.standard_normal(size))
    start_time = pd.Timestamp("2020-01-01 00:00:00")
    timestamps = pd.date_range(start_time, periods=n, freq="min")
    data = pd.DataFrame({"timestamp": timestamps, "target": targets})
    return data

def calc_phi_from_fve(fve: float) -> float:
    if fve < 0 or fve >= 1:
        raise ValueError("fve must be in [0, 1).")
    phi = np.sqrt(fve)
    return phi

# Fill in the function below. Assume that
#
# - The `"target"` column of `test_data` contains the data.
# - `predictions` contains predictions for the last `prediction_length` observations in `test_data`.
# - The next-to-last column of `predictions` contains lower bounds for prediction intervals and the last column contains upper bounds.
#
# The function should return a `Series` whose kth entry is `True` if and only if the kth prediction is between the corresponding bounds,
# inclusive. The index values of the `Series` should be the strings `"1"`, ..., `"f{prediction_length}"`.
def help_check_prediction_intervals(test_data: TimeSeriesDataFrame, prediction_length: int, predictions: TimeSeriesDataFrame) -> pd.Series:
    actuals = test_data["target"].iloc[-prediction_length: ].values
    upper_bounds = predictions.iloc[:,-1].values
    lower_bounds = predictions.iloc[:,-2].values
    
    coverage = (actuals >= lower_bounds) & (actuals <= upper_bounds)
    return pd.Series(coverage, index=[str(i + 1) for i in range(prediction_length)])

# Fill in the function below. Assume that
#
# - The `"timestamp"` column of `data` contains observation times and the `"target"` column contains observation values.
# - The last `prediction_length` observations in `data` are test observations and the prior observations are training observations.
# - Hyperparameter tuning on a validation set is to be done using `eval_metric`.
# - Level-`ci_level` prediction intervals are needed. `ci_level` could be 0.95, for example.
# - The training of any one model cannot take longer than `time_limit` seconds.
# - The models to be used are specified in `hyperparameters`. For example, `hyperparameters` could equal `{"AutoARIMA": {}}`.
#
# For each model, the function should fit the model, calculate prediction intervals, and then use `help_check_prediction_intervals` to check
# whether the intervals contain the corresponding observations. The function should return a `DataFrame` with one row for each model. In the row
# for a model, the first column should contain the model name and the next `prediction_length` columns should contain flags indicating whether
# the intervals contain the corresponding observations. The column names should be `"model"`, `"1"`, ..., `f"{prediction_length}"`.
def check_prediction_intervals(
        data: pd.DataFrame,
        prediction_length: int,
        eval_metric: str,
        ci_level: float,
        time_limit: int,
        hyperparameters: Dict[str | Type, Any]
    ) -> pd.DataFrame:
    
    train_data = data.iloc[:-prediction_length]
    test_data = data.iloc[-prediction_length:]
    
    lower_quantile = (1 - ci_level) / 2
    upper_quantile = 1 - (1 - ci_level) / 2
    quantiles = [lower_quantile, upper_quantile]
    
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        eval_metric=eval_metric,
        path="model_store", 
        quantile_levels=quantiles
    )
    predictor.fit(train_data, hyperparameters=hyperparameters, time_limit=time_limit, presets="best_quality", verbosity=4)
    
    results = []
    for model_name in predictor.model_names():
        model_predictions = predictor.predict(data=test_data, model=model_name)
        coverage_flags = help_check_prediction_intervals(test_data, prediction_length, model_predictions)
        results.append([model_name] + coverage_flags.tolist())
    
    columns = ["model"] + [str(i + 1) for i in range(prediction_length)]
    return pd.DataFrame(results, columns=columns)
    

# Fill in the function below. Assume that
#
# - The number of runs to execute is `num_runs`.
# - Datasets should be generated from an AR(1) model whose fraction of variance explained (FVE) is `fve`.
# - The training part of a dataset should contain `train_size` observations.
# - Predictions are to computed for the last `prediction_length` observations in a dataset.
# - Hyperparameter tuning on a validation set is to be done using `eval_metric`.
# - Level-`ci_level` prediction intervals are needed. `ci_level` could be 0.95, for example.
# - The training of any one model cannot take longer than `time_limit` seconds.
# - The models to be used are specified in `hyperparameters`. For example, `hyperparameters` could equal `{"AutoARIMA": {}}`.
#
# The function should generate `num_runs` datasets; for each dataset, it should use `check_prediction_intervals` to fit the models to the
# dataset, compute prediction intervals, and check whether the intervals contain the corresponding observations. The function should return a
# `DataFrame` that is the concatenation of the `DataFrames` produced by `check_prediction_intervals`. The first column should be a `"run_num"`
# column that gives the run number for each `DataFrame` output by `check_prediction_intervals`.
def run_ar1_experiment(
        num_runs: int,
        fve: float,
        train_size: int,
        prediction_length: int,
        eval_metric: str,
        ci_level: float,
        time_limit: int,
        hyperparameters: Dict[str | Type, Any],
        rng: np.random.Generator
    ) -> pd.DataFrame:
    
    all_results = []
    
    for run in range(num_runs):
        phi = calc_phi_from_fve(fve=fve)
        data = simulate_ar1(phi=phi, sigma=1, n=train_size, rng=rng) ## Should sigma be a parameter?
        data["item_id"] = 0
        result_df = check_prediction_intervals(data, prediction_length=prediction_length, eval_metric=eval_metric, ci_level=ci_level, time_limit=time_limit, hyperparameters=hyperparameters)
        result_df.insert(0, "run_num", run+1)
        all_results.append(result_df)
    
    return pd.concat(all_results, ignore_index=True)

def make_dir_name(
    num_runs: int,
    fve: float,
    train_size: int,
    prediction_length: int,
    eval_metric: str,
    ci_level: float,
    time_limit: int
    ) -> str:
    dir_name = f"{num_runs}_{fve}_{train_size}_{prediction_length}_{eval_metric}_{ci_level}_{time_limit}"
    return dir_name

# Fill in the function below. Assume that
#
# - `results` contains the output of `run_ar1_experiment`.
# - The number of runs to execute is `num_runs`.
# - Datasets should be generated from an AR(1) model whose fraction of variance explained (FVE) is `fve`.
# - The training part of a dataset should contain `train_size` observations.
# - Predictions are to computed for the last `prediction_length` observations in a dataset.
# - Hyperparameter tuning on a validation set is to be done using `eval_metric`.
# - Level-`ci_level` prediction intervals are needed. `ci_level` could be 0.95, for example.
# - The training of any one model cannot take longer than `time_limit` seconds.
#
# For each model, for h = 1, ..., `prediction_length`, the function should compute the coverage, i.e., the proportion of h-step-ahead
# prediction intervals that contained the corresponding observation. The function should then output a plot that has one panel for each model,
# with the panel for a model plotting the coverage versus h. Use both `geom_point` and `geom_line`. Use `make_dir_name` to make the title.
def plot_results(
        results: pd.DataFrame,
        num_runs: int,
        fve: float,
        train_size: int,
        prediction_length: int,
        eval_metric: str,
        ci_level: float,
        time_limit: int
    ) -> ggplot:
    
    melted_results = results.melt(id_vars=["run_num", "model"], var_name="h", value_name="coverage")
    coverage_df = melted_results.groupby(["model", "h"]).mean().reset_index()
    coverage_df["h"] = coverage_df["h"].astype(int)
    plot = (
        ggplot(coverage_df, aes(x="h", y="coverage", group="model", color="model")) +
        geom_point() +
        geom_line() +
        facet_wrap("~model") +
        labs(
            title=make_dir_name(num_runs, fve, train_size, prediction_length, eval_metric, ci_level, time_limit),
            x="Prediction Horizon (h)",
            y="Coverage Proportion"
        )
    )
    
    return plot

if __name__ == "__main__":
    print("Running AR(1) Experiment", end="", flush=True)
    experiment_start_time = time.time()

    ################################################################################
    # Parse the command-line arguments
    ################################################################################

    parser = argparse.ArgumentParser(description="Run an AR(1) experiment")
    parser.add_argument("--num_runs", type=int, help="Number of runs to execute")
    parser.add_argument("--fve", type=float, help="Fraction of variance explained of the AR(1) model, e.g., 0.5")
    parser.add_argument("--train_size", type=int, help="Size of a training set")
    parser.add_argument("--prediction_length", type=int, help="Number of time steps after a training set to compute predictions at")
    parser.add_argument("--eval_metric", type=str, help="Metric to use for hyperparameter tuning on a validation set")
    parser.add_argument("--ci_level", type=float, help="Level of the prediction intervals, e.g., 0.95")
    parser.add_argument("--time_limit", type=int, help="Maximum number of seconds for fitting a model")
    parser.add_argument("--epochs", type=int, help="Number of Epochs for fitting a model")

    cmd_args = parser.parse_args()
    num_runs = cmd_args.num_runs
    fve = cmd_args.fve
    train_size = cmd_args.train_size
    prediction_length = cmd_args.prediction_length
    eval_metric = cmd_args.eval_metric
    ci_level = cmd_args.ci_level
    time_limit = cmd_args.time_limit
    epochs = cmd_args.epochs

    ################################################################################
    # Run the experiment
    ################################################################################

    hyperparameters = {#"AutoARIMA": {}, 
                       #"PatchTST": {"max_epochs": epochs}
                       "TemporalFusionTransformer": {"max_epochs": epochs}
                       }
    rng = np.random.default_rng(12345)
    results = run_ar1_experiment(num_runs, fve, train_size, prediction_length, eval_metric, ci_level, time_limit, hyperparameters, rng)
    results_plot = plot_results(results, num_runs, fve, train_size, prediction_length, eval_metric, ci_level, time_limit)

    ################################################################################
    # Save the results
    ################################################################################

    dir_name = make_dir_name(num_runs, fve, train_size, prediction_length, eval_metric, ci_level, time_limit)
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)

    results.to_parquet(os.path.join(dir_name, "results.parquet"))
    save_as_pdf_pages([results_plot], os.path.join(dir_name, "results_plot.pdf"))

    experiment_elapsed_time = time.time() - experiment_start_time
    print(f"\rRunning AR(1) experiment ({int(experiment_elapsed_time)}s)", flush=True)
    print("Done")
