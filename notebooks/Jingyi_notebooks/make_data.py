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

def generate_arima_data(fve: float, num_runs: int, train_size: int, prediction_length: int, seed: int) -> list[pd.DataFrame]:
    np.random.seed(seed)  # Set random seed for replication
    rng = np.random.default_rng(seed)

    # Calculate phi from FVE
    phi = calc_phi_from_fve(fve)
    sigma = 1.0  # Standard deviation for the noise

    # Store the datasets for training and testing
    train_data = []
    test_data = []

    for run in range(num_runs):
        # Generate AR1 time series data
        data = simulate_ar1(phi=phi, sigma=sigma, n=train_size + prediction_length, rng=rng)
        
        # Split the data into training and test sets
        train_df = data.iloc[:train_size]
        test_df = data.iloc[train_size:]

        train_data.append(train_df)
        test_data.append(test_df)
        train_data["run_idx"] = run
        train_data["fve"] = fve
        test_data["run_idx"] = run
        test_data["fve"] = fve

    # Combine all runs into single train and test dataframes
    full_train_df = pd.concat(train_data, ignore_index=True)
    full_test_df = pd.concat(test_data, ignore_index=True)
    full_train_df.insert(0, "item_id", 0)
    full_test_df.insert(0, "item_id", 0)

    return full_train_df, full_test_df

if __name__ == "__main__":
    print("Generate ARIMA Dataset")

    ################################################################################
    # Parse the command-line arguments
    ################################################################################

    parser = argparse.ArgumentParser(description="Generate an ARIMA Dataset")
    parser.add_argument("--num_runs", type=int, required=True, help="Number of runs to execute")
    parser.add_argument("--fve", type=float, required=True, help="Fraction of variance explained of the AR(1) model, e.g., 0.5")
    parser.add_argument("--train_size", type=int, required=True, help="Size of a training set")
    parser.add_argument(
        "--prediction_length", type=int, required=True, help="Number of time steps after a training set to compute predictions at"
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="seed"
    )

    cmd_args = parser.parse_args()
    num_runs = cmd_args.num_runs # number of replications
    fve = cmd_args.fve
    train_size = cmd_args.train_size
    prediction_length = cmd_args.prediction_length
    seed = cmd_args.seed

    ################################################################################
    # Run the generator
    ################################################################################
    full_train_df, full_test_df = generate_arima_data(fve, num_runs, train_size, prediction_length, seed)

    ################################################################################
    # Save the data
    ################################################################################
    dir_name = "../data"
    # if os.path.exists(dir_name):
    #     shutil.rmtree(dir_name)
    # os.mkdir(dir_name)
    
    full_train_df.to_parquet(f'../data/train_data.parquet', index=False)
    full_test_df.to_parquet(f'../data/test_data.parquet', index=False)
    print(f"Shape of the full train dataframe: {full_train_df.shape}")
    print(f"Shape of the full test dataframe: {full_test_df.shape}")
    print(f"Data generation complete. Training and testing datasets are saved in ../data")