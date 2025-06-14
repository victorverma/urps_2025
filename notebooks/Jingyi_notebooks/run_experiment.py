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
    # Extract the actual values for the last `prediction_length` steps
    obs_data = test_data["target"].iloc[-prediction_length:]

    # Identify the lower- and upper-bound columns
    lower_bounds = predictions.iloc[:, -2]
    upper_bounds = predictions.iloc[:, -1]

    # Check coverage, inclusive of the bounds
    coverage = (obs_data >= lower_bounds) & (obs_data <= upper_bounds)

    # Rename the index to string "1", ..., f"{prediction_length}"
    coverage.index = [str(i) for i in range(1, prediction_length + 1)]
    
    return coverage
    
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
    
    # For time series, add an 'item_id' = 0
    data["item_id"] = 0

    # Assume data is already modified with timestamp, target columns. Convert into AutoGluon TimeSeriesDataFrame objects.
    data = TimeSeriesDataFrame.from_data_frame(data)

    # Separate training and test sets
    train_df, test_df = data.train_test_split(prediction_length)
    
    # Prepare for storage of results
    results_list = []

    # Compute the lower/upper quantiles
    alpha = 1.0 - ci_level   # e.g., if ci_level=0.95, alpha=0.05
    lower_quantile = alpha / 2.0
    upper_quantile = 1.0 - lower_quantile
    
    for model_name, model_hparams in hyperparameters.items():
        # Create a fresh predictor path to avoid collisions
        # create the predictor
        predictor = TimeSeriesPredictor(
            prediction_length = prediction_length,
            eval_metric = eval_metric,
            quantile_levels = [lower_quantile, upper_quantile]
        )
        
        # Fit the model
        predictor.fit(
            train_data=train_df,
            hyperparameters={model_name: model_hparams},
            time_limit=time_limit
        )
        
        # Generate predictions with intervals
        pred_ints = predictor.predict(train_df)
        
        # The "next-to-last" column is the lower bound, the last column is upper bound
        coverage_flags = help_check_prediction_intervals(
            test_df,
            prediction_length,
            pred_ints
        )
        
        # Build one row for this model
        row = {"model": model_name}
        for i in range(1, prediction_length + 1):
            row[str(i)] = coverage_flags[str(i)]
        results_list.append(row)
    
    results_df = pd.DataFrame(results_list)
    return results_df

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
    results_list = []
    
    # Compute AR(1) parameter phi from fraction of variance explained
    phi = calc_phi_from_fve(fve)
    
    for run_idx in range(1, num_runs + 1):
        # Simulate a new dataset of length train_size + prediction_length
        n = train_size + prediction_length
        data = simulate_ar1(phi=phi, sigma=1.0, n=n, rng=rng)
        
        # Check coverage for each model
        coverage_df = check_prediction_intervals(
            data=data,
            prediction_length=prediction_length,
            eval_metric=eval_metric,
            ci_level=ci_level,
            time_limit=time_limit,
            hyperparameters=hyperparameters
        )
        
        # Tag with run number
        coverage_df["run_num"] = run_idx
        results_list.append(coverage_df)
    
    # Concatenate all
    all_results = pd.concat(results_list, ignore_index=True)
    # Put 'run_num' in front
    cols = ["run_num", "model"] + [str(i) for i in range(1, prediction_length + 1)]
    all_results = all_results[cols]
    return all_results

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
    # results columns: ["run_num", "model", "1", "2", ..., "prediction_length"]
    # Convert True/False to numeric coverage (1/0)
    coverage_cols = [str(i) for i in range(1, prediction_length + 1)]
    
    # Melt into long form for plotting:
    long_df = results.melt(
        id_vars=["run_num", "model"],
        value_vars=coverage_cols,
        var_name="horizon",
        value_name="contains"
    )
    
    # Convert bool -> float (0/1) if necessary
    long_df["contains"] = long_df["contains"].astype(float)
    # Compute coverage by (model, horizon)
    coverage_df = long_df.groupby(["model", "horizon"], as_index=False)["contains"].mean()
    coverage_df.rename(columns={"contains": "coverage"}, inplace=True)
    
    # Convert horizon to numeric so ggplot can treat it properly on x-axis
    coverage_df["horizon"] = coverage_df["horizon"].astype(int)
    
    # Build the plot
    plot_title = make_dir_name(num_runs, fve, train_size, prediction_length, eval_metric, ci_level, time_limit)
    p = (
        ggplot(coverage_df, aes(x="horizon", y="coverage", group="model")) +
        geom_point() +
        geom_line() +
        facet_wrap("~model", scales="fixed") +
        labs(title=plot_title, x="Horizon (h)", y="Coverage") +
        theme(subplots_adjust={'wspace': 0.25})
    )
    return p

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

    cmd_args = parser.parse_args()
    num_runs = cmd_args.num_runs
    fve = cmd_args.fve
    train_size = cmd_args.train_size
    prediction_length = cmd_args.prediction_length
    eval_metric = cmd_args.eval_metric
    ci_level = cmd_args.ci_level
    time_limit = cmd_args.time_limit

    ################################################################################
    # Run the experiment
    ################################################################################

    hyperparameters = {"AutoARIMA": {}, "PatchTST": {}, "TemporalFusionTransformer": {}}
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
