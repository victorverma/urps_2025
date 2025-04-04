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


def rmse(y_true, y_pred):
    """Compute RMSE (Root Mean Squared Error)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def avg_pred_length(lower_bounds, upper_bounds):
    """Compute Average Prediction Length (Interval Width)."""
    return np.mean(upper_bounds - lower_bounds)

def help_check_prediction_intervals(test_data: TimeSeriesDataFrame, prediction_length: int, predictions: TimeSeriesDataFrame) -> pd.Series:
    observed = test_data["target"]
    are_lower_bounds_right = observed >= predictions.iloc[:, -2]
    are_upper_bounds_right = observed <= predictions.iloc[:, -1]
    checks = are_lower_bounds_right & are_upper_bounds_right
    checks.index = [str(i) for i in range(1, prediction_length + 1)]
    return checks

def train_once(
        train_data: pd.DataFrame,
        prediction_length: int,
        eval_metric: str,
        ci_level: float,
        time_limit: int,
        verbosity: int,
        hyperparameters: Dict[str | Type, Any]
    ) -> TimeSeriesPredictor:

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        eval_metric=eval_metric,
        verbosity=verbosity,
        quantile_levels=[(1 - ci_level) / 2, (1 + ci_level) / 2],
        freq="min"
    )

    predictor.fit(train_data=train_data, time_limit=time_limit, hyperparameters=hyperparameters, enable_ensemble=False)

    return predictor

def check_prediction_intervals(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        prediction_length: int,
        predictor: TimeSeriesPredictor,
        hyperparameters: Dict[str | Type, Any]
    ) -> pd.DataFrame:

    results_list = []
    for model_name, _ in hyperparameters.items():
        predictions = predictor.predict(train_df, model=model_name)
        lower_bounds, upper_bounds = predictions.iloc[:, -2], predictions.iloc[:, -1]

        coverage_flags = help_check_prediction_intervals(test_df, prediction_length, predictions)

        df = pd.DataFrame(
            {"model_name": model_name,
            "h": range(1, prediction_length+1),
            "observed": test_df["target"].values,
            "prediction": predictions["mean"].values, 
            "lower_bound": lower_bounds.values, 
            "upper_bound": upper_bounds.values,
            "coverage_flags": coverage_flags.values, 
            "squared_error": (test_df["target"] - predictions.iloc[:, 0]) ** 2
            }        
        )
        df.reset_index(drop=True, inplace=True)

        results_list.append(df)

    return pd.concat(results_list)


def do_1_run(
        run_num: int, # the ith index from range(1, num_runs)
        train_size: int,
        prediction_length: int,
        predictor: TimeSeriesPredictor,
        train_full_df: pd.DataFrame,
        test_full_df: pd.DataFrame,
        hyperparameters: Dict[str | Type, Any]
    ) -> pd.DataFrame:

    # extract the one dataset for each testing 
    train_start_idx = run_num * train_size
    train_end_idx = train_start_idx + train_size
    train_df = train_full_df.iloc[train_start_idx:train_end_idx, ]
    
    test_start_idx = run_num * prediction_length
    test_end_idx = test_start_idx + prediction_length
    test_df = test_full_df.iloc[test_start_idx:test_end_idx, ]

    checks = check_prediction_intervals(train_df, test_df, prediction_length, predictor, hyperparameters)
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

    # load in full train and test data
    train_full_df = "../data/train_data.parquet"
    test_full_df = "../data/test_data.parquet"
    train_full_df = TimeSeriesDataFrame.from_data_frame(train_full_df)
    test_full_df = TimeSeriesDataFrame.from_data_frame(test_full_df)

    # train the model once with given data
    train_once_df = train_full_df.iloc[0:train_size,]

    predictor = train_once(train_once_df, prediction_length, eval_metric, ci_level, time_limit, verbosity, hyperparameters)

    # predict different sets of testing data based on the only trained model
    if max_workers == 1:
        results = [
            do_1_run(run_num, train_size, prediction_length, predictor, train_full_df, test_full_df, hyperparameters)
            for run_num in range(num_runs)
        ]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    do_1_run,
                    range(num_runs),
                    [train_size] * num_runs,
                    [prediction_length] * num_runs,
                    [predictor] * num_runs,
                    [train_full_df] * num_runs,
                    [test_full_df] * num_runs,
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
    dir_name = f"train_once_{num_runs}_{fve}_{train_size}_{prediction_length}_{eval_metric}_{ci_level}_{time_limit}_{max_epochs}"
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

    df = results.groupby(["model_name", "h"], as_index=False).agg(
        coverage_accuracy=("coverage_flags", "mean"),  # Proportion of times coverage is True
        mean_squared_error=("squared_error", "mean"),   # Mean squared error for each group
        RMSE=("squared_error", lambda x: np.sqrt(x.mean()))
    )

    # Coverage accuracy plot
    p_coverage = (
        ggplot(df, aes(x="h", y="coverage_accuracy", color="model_name")) +
        geom_point() + geom_line() +
        facet_wrap("~model_name", scales="fixed") +  # Separate plots for each model
        labs(title="Coverage Accuracy by Model and Horizon", x="Horizon (h)", y="Coverage Accuracy") +
        theme(subplots_adjust={'wspace': 0.25})
    )

    # Squared error plot
    p_error = (
        ggplot(df, aes(x="h", y="RMSE", color="model_name")) +
        geom_point() + geom_line() +
        facet_wrap("~model_name", scales="fixed") +  # Separate plots for each model
        labs(title="RMSE by Model and Horizon", x="Horizon (h)", y="RMSE") +
        theme(subplots_adjust={'wspace': 0.25})
    )

    return [p_coverage, p_error]


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
        "AutoARIMA": {}, 
        "PatchTST": {'max_epochs': max_epochs, 'patch_len': 1, 'stride': 4, 'nhead': 1, 'd_model': 1, 'context_length': 8},
        # "TemporalFusionTransformer": {"max_epochs": max_epochs, 'context_length': 8}
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
