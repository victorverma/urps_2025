import argparse
import numpy as np
import os
import pandas as pd
import shutil
from statsmodels.tsa.arima_process import ArmaProcess


def simulate_ar1(phi: float, sigma: float, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulate an AR(1) time series process.
    
    Args:
        phi: Autoregressive parameter (must have absolute value < 1)
        sigma: Standard deviation of the noise
        n: Number of time steps to generate
        rng: Random number generator instance
        
    Returns:
        DataFrame with columns 'timestamp' and 'target' containing the generated time series
        
    Raises:
        ValueError: If phi has absolute value >= 1
    """
    if np.abs(phi) >= 1:
        raise ValueError("The absolute value of phi must be less than one.")
    arma_process = ArmaProcess(ar=np.array([1, -phi]), ma = np.array([1]))
    targets = arma_process.generate_sample(n, sigma, distrvs=lambda size: rng.standard_normal(size))
    start_time = pd.Timestamp("2020-01-01 00:00:00")
    timestamps = pd.date_range(start_time, periods=n, freq="min")
    data = pd.DataFrame({"timestamp": timestamps, "target": targets})
    return data

def calc_phi_from_fve(fve: float) -> float:
    """
    Calculate the AR(1) coefficient phi from Fraction of Variance Explained (FVE).
    
    Args:
        fve: Fraction of variance explained (must be in [0, 1))
        
    Returns:
        The calculated phi value
        
    Raises:
        ValueError: If fve is outside [0, 1)
    """
    if fve < 0 or fve >= 1:
        raise ValueError("fve must be in [0, 1).")
    phi = np.sqrt(fve)
    return phi

def generate_ar1_data(fve: float, num_runs: int, train_size: int, prediction_length: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate training and test datasets for AR(1) time series experiments.
    
    Args:
        fve: Fraction of variance explained for the AR(1) process
        num_runs: Number of independent time series to generate
        train_size: Length of training period
        prediction_length: Length of test period
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing (full_train_df, full_test_df) DataFrames with columns:
        - item_id: Constant 0 (single time series)
        - timestamp: Datetime index
        - target: Generated values
        - run_idx: Identifier for each independent run
        - fve: Fraction of variance explained (repeated for each row)
    """
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
        train_df = data.iloc[:train_size].copy()
        test_df = data.iloc[train_size:].copy()

        # Add metadata columns
        train_data["run_idx"] = run
        train_data["fve"] = fve
        test_data["run_idx"] = run
        test_data["fve"] = fve

        train_data.append(train_df)
        test_data.append(test_df)

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
    full_train_df, full_test_df = generate_ar1_data(fve, num_runs, train_size, prediction_length, seed)

    ################################################################################
    # Save the data
    ################################################################################
    dir_name = "../data"
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)
    
    full_train_df.to_parquet(f'../data/train_data.parquet', index=False)
    full_test_df.to_parquet(f'../data/test_data.parquet', index=False)
    print(f"Shape of the full train dataframe: {full_train_df.shape}")
    print(f"Shape of the full test dataframe: {full_test_df.shape}")
    print(f"Data generation complete. Training and testing datasets are saved in ../data")