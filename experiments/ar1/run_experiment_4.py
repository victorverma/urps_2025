import argparse
import numpy as np
import os
import pandas as pd
import shutil
import time
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from concurrent.futures import ProcessPoolExecutor
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

def rmse(y_true, y_pred):
    """Compute RMSE (Root Mean Squared Error)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def avg_pred_length(lower_bounds, upper_bounds):
    """Compute Average Prediction Length (Interval Width)."""
    return np.mean(upper_bounds - lower_bounds)

def help_check_prediction_intervals(test_data: TimeSeriesDataFrame, prediction_length: int, predictions: TimeSeriesDataFrame) -> pd.Series:
    actuals = test_data["target"].iloc[-prediction_length:]
    are_lower_bounds_right = actuals >= predictions.iloc[:, -2]
    are_upper_bounds_right = actuals <= predictions.iloc[:, -1]
    checks = are_lower_bounds_right & are_upper_bounds_right
    checks.index = [str(i) for i in range(1, prediction_length + 1)]
    return checks

import torch

def check_prediction_intervals(
        data: pd.DataFrame,
        prediction_length: int,
        eval_metric: str,
        ci_level: float,
        time_limit: int,
        verbosity: int,
        hyperparameters: Dict[str | Type, Any]
    ) -> pd.DataFrame:

    data["item_id"] = 0
    data = TimeSeriesDataFrame(data)

    train_df, test_df = data.train_test_split(prediction_length)

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        eval_metric=eval_metric,
        verbosity=verbosity,
        quantile_levels=[(1 - ci_level) / 2, (1 + ci_level) / 2]
    )

    start_time = time.time()
    predictor.fit(train_data=train_df, time_limit=time_limit, hyperparameters=hyperparameters, enable_ensemble=False)
    training_runtime = time.time() - start_time

    leaderboard_df = predictor.leaderboard(extra_info=True, display=False)

    results_list = []
    for model_name, _ in hyperparameters.items():
        predictions = predictor.predict(train_df, model=model_name)
        lower_bounds, upper_bounds = predictions.iloc[:, -2], predictions.iloc[:, -1]

        coverage_flags = help_check_prediction_intervals(test_df, prediction_length, predictions)

        print(len(predictions["mean"]))
        print(len(lower_bounds))
        print(len(upper_bounds))
        print(len(coverage_flags))
        print(len((test_df["target"].iloc[-prediction_length:] - predictions.iloc[:, 0]) ** 2))
        df = pd.DataFrame(
            {"model_name": model_name,
            "h": range(1, prediction_length+1),
            "observed":test_df["target"].iloc[-prediction_length:],
                "prediction": predictions["mean"].values, "lower_bound": lower_bounds.values, "upper_bound": upper_bounds.values,
        "coverage_flags": coverage_flags.values, 
        "squared_error": (test_df["target"].iloc[-prediction_length:] - predictions.iloc[:, 0]) ** 2
        }        
        )
        df.reset_index(drop=True, inplace=True)
        print(df)

        #row = {"model": model_name}
        # for i in range(1, prediction_length + 1):
        #     row[str(i)] = coverage_flags[str(i)]


        # row["RMSE"] = np.sqrt(np.mean((test_df["target"].iloc[-prediction_length:] - predictions.iloc[:, 0]) ** 2))
        # row["Avg_Pred_Length"] = np.mean(upper_bounds - lower_bounds)
        # row["Training_Loss"] = float(predictor.fit_summary()["model_performance"]["PatchTST"])
        # row["Runtime"] = float(predictor.fit_summary()["model_fit_times"]["PatchTST"])

        results_list.append(df)

    return pd.concat(results_list)


def do_1_run(
        run_num: int,
        phi: float,
        sigma: float,
        n: int,
        prediction_length: int,
        eval_metric: str,
        ci_level: float,
        time_limit: int,
        verbosity: int,
        hyperparameters: Dict[str | Type, Any]
    ) -> pd.DataFrame:
    rng = np.random.default_rng(run_num)
    data = simulate_ar1(phi, sigma, n, rng)
    checks = check_prediction_intervals(data, prediction_length, eval_metric, ci_level, time_limit, verbosity, hyperparameters)
    checks.insert(0, "run_num", run_num)
    return checks

def run_experiment(
        num_runs: int,
        fve: float,
        train_size: int,
        prediction_length: int,
        eval_metric: str,
        ci_level: float,
        time_limit: int,
        verbosity: int,
        hyperparameters: Dict[str | Type, Any],
        max_workers: int
    ) -> pd.DataFrame:
    phi = calc_phi_from_fve(fve)
    sigma = 1
    n = train_size + prediction_length

    if max_workers == 1:
        results = [
            do_1_run(run_num, phi, sigma, n, prediction_length, eval_metric, ci_level, time_limit, verbosity, hyperparameters)
            for run_num in range(num_runs)
        ]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    do_1_run,
                    range(num_runs),
                    [phi] * num_runs,
                    [sigma] * num_runs,
                    [n] * num_runs,
                    [prediction_length] * num_runs,
                    [eval_metric] * num_runs,
                    [ci_level] * num_runs,
                    [time_limit] * num_runs,
                    [verbosity] * num_runs,
                    [hyperparameters] * num_runs
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
    dir_name = f"{num_runs}_{fve}_{train_size}_{prediction_length}_{eval_metric}_{ci_level}_{time_limit}_{max_epochs}"
    return dir_name

def plot_results(
        results: pd.DataFrame,
        num_runs: int,
        fve: float,
        train_size: int,
        prediction_length: int,
        eval_metric: str,
        ci_level: float,
        time_limit: int,
        max_epochs: int
    ) -> list[ggplot]:

    coverage_cols = [str(i) for i in range(1, prediction_length + 1)]

    long_df = results.melt(
        id_vars=["run_num", "model"],
        value_vars=coverage_cols,
        var_name="horizon",
        value_name="contains"
    )
    
    long_df["contains"] = long_df["contains"].astype(float)
    coverage_df = long_df.groupby(["model", "horizon"], as_index=False)["contains"].mean()
    coverage_df.rename(columns={"contains": "coverage"}, inplace=True)
    coverage_df["horizon"] = coverage_df["horizon"].astype(int)

    results["RMSE"] = pd.to_numeric(results["RMSE"], errors="coerce")
    results["Avg_Pred_Length"] = pd.to_numeric(results["Avg_Pred_Length"], errors="coerce")
    results["Training_Loss"] = pd.to_numeric(results["Training_Loss"], errors="coerce")
    results["Runtime"] = pd.to_numeric(results["Runtime"], errors="coerce")

    plot_title = make_dir_name(num_runs, fve, train_size, prediction_length, eval_metric, ci_level, time_limit, max_epochs)

    p_coverage = (
        ggplot(coverage_df, aes(x="horizon", y="coverage", group="model")) +
        geom_point() + geom_line() +
        facet_wrap("~model", scales="fixed") +
        labs(title=plot_title, x="Horizon (h)", y="Coverage") +
        theme(subplots_adjust={'wspace': 0.25})
    )

    p_rmse = (
        ggplot(results.reset_index(), aes(x="index", y="RMSE", group="model", color="model")) +
        geom_point() + geom_line() +
        labs(title="RMSE (h)", x="Run Index", y="Root Mean Squared Error") +
        theme(axis_text_x=element_text(rotation=45, hjust=1))
    )

    p_pred_length = (
        ggplot(results.reset_index(), aes(x="index", y="Avg_Pred_Length", group="model", color="model")) +
        geom_point() + geom_line() +
        labs(title="Avg Prediction Length (h)", x="Run Index", y="Prediction Interval Width") +
        theme(axis_text_x=element_text(rotation=45, hjust=1))
    )

    p_training_loss = (
        ggplot(results.reset_index(), aes(x="index", y="Training_Loss", group="model", color="model")) +
        geom_point() + geom_line() +
        labs(title="Training Loss", x="Run Index", y="Loss Value") +
        theme(axis_text_x=element_text(rotation=45, hjust=1))
    )

    p_runtime = (
        ggplot(results.reset_index(), aes(x="index", y="Runtime", group="model", color="model")) +
        geom_point() + geom_line() +
        labs(title="Runtime (s)", x="Run Index", y="Execution Time (Seconds)") +
        theme(axis_text_x=element_text(rotation=45, hjust=1))
    )

    return [p_coverage, p_rmse, p_pred_length, p_training_loss, p_runtime]


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

    hyperparameters = {
        # "AutoARIMA": {}, 
     "PatchTST": {'max_epochs': max_epochs, 'patch_len': 1, 'stride': 4, 'nhead': 1, 'd_model': 1, 'context_length': 8}
    # "TemporalFusionTransformer": {"max_epochs": max_epochs}
    }
    results = run_experiment(
        num_runs, fve, train_size, prediction_length, eval_metric, ci_level, time_limit, verbosity, hyperparameters, max_workers
    )
    results_plot = plot_results(results, num_runs, fve, train_size, prediction_length, eval_metric, ci_level, time_limit, max_epochs)

    ################################################################################
    # Save the results
    ################################################################################

    dir_name = make_dir_name(num_runs, fve, train_size, prediction_length, eval_metric, ci_level, time_limit, max_epochs)
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)

    results.to_parquet(os.path.join(dir_name, "results.parquet"))
    for i, plot in enumerate(results_plot):
        plot.save(os.path.join(dir_name, f"plot_{i}.pdf"))

    experiment_elapsed_time = time.time() - experiment_start_time
    print(f"Done ({int(experiment_elapsed_time)}s)")