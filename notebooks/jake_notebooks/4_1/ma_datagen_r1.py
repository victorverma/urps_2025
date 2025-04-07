import argparse
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
import os


def generate_ma_series(ma_coefs: np.ndarray, total_length: int, seed: int) -> pd.DataFrame:
    ar_coefs = np.array([1.0])
    ma_process = ArmaProcess(ar_coefs, ma_coefs)
    np.random.seed(seed)
    values = ma_process.generate_sample(nsample=total_length)
    timestamps = pd.date_range(start="2020-01-01", periods=total_length, freq="min")
    return pd.DataFrame({"timestamp": timestamps, "target": values})


def generate_exponential_datasets(q, thetas, train_size, prediction_length, num_runs, seed):
    train_rows = []
    test_rows = []
    total_length = train_size + prediction_length
    rng = np.random.default_rng(seed)

    for theta in thetas:
        for run in range(num_runs):
            ui = rng.uniform(-1, 1, q)
            ma_coefs = np.array([1.0] + list(ui * (theta ** np.arange(1, q + 1))))
            df = generate_ma_series(ma_coefs, total_length, seed + run)
            df["run_idx"] = run
            df["theta"] = theta
            df["method"] = "exponential"
            df["item_id"] = f"exp_{theta}_{run}"
            train_rows.append(df.iloc[:train_size])
            test_rows.append(df.iloc[train_size:])

    return pd.concat(train_rows, ignore_index=True), pd.concat(test_rows, ignore_index=True)


def generate_powerlaw_datasets(q, gammas, train_size, prediction_length, num_runs, seed):
    train_rows = []
    test_rows = []
    total_length = train_size + prediction_length

    for gamma in gammas:
        for run in range(num_runs):
            ma_coefs = np.array([1.0] + list(1 / (np.arange(1, q + 1) ** gamma)))
            df = generate_ma_series(ma_coefs, total_length, seed + run)
            df["run_idx"] = run
            df["gamma"] = gamma
            df["method"] = "powerlaw"
            df["item_id"] = f"power_{gamma}_{run}"
            train_rows.append(df.iloc[:train_size])
            test_rows.append(df.iloc[train_size:])

    return pd.concat(train_rows, ignore_index=True), pd.concat(test_rows, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MA datasets (exponential or power-law)")
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--train_size", type=int, required=True)
    parser.add_argument("--prediction_length", type=int, required=True)
    parser.add_argument("--num_runs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--thetas", type=float, nargs="*", help="Theta values for exponential MA")
    parser.add_argument("--gammas", type=float, nargs="*", help="Gamma values for power-law MA")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory for parquet files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.thetas:
        train_exp, test_exp = generate_exponential_datasets(
            q=args.q,
            thetas=args.thetas,
            train_size=args.train_size,
            prediction_length=args.prediction_length,
            num_runs=args.num_runs,
            seed=args.seed
        )
        train_exp.to_parquet(os.path.join(args.output_dir, "train_exp.parquet"), index=False)
        test_exp.to_parquet(os.path.join(args.output_dir, "test_exp.parquet"), index=False)
        print(f"Saved exponential MA train/test datasets to {args.output_dir}")

    if args.gammas:
        train_pow, test_pow = generate_powerlaw_datasets(
            q=args.q,
            gammas=args.gammas,
            train_size=args.train_size,
            prediction_length=args.prediction_length,
            num_runs=args.num_runs,
            seed=args.seed
        )
        train_pow.to_parquet(os.path.join(args.output_dir, "train_powerlaw.parquet"), index=False)
        test_pow.to_parquet(os.path.join(args.output_dir, "test_powerlaw.parquet"), index=False)
        print(f"Saved power-law MA train/test datasets to {args.output_dir}")
