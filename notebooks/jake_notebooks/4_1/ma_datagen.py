import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
import argparse

def generate_exp_decay_with_ui(q, thetas, n, num_datasets):
    all_data = []
    for theta in thetas:
        for i in range(num_datasets):
            ui = np.random.uniform(-1, 1, q)
            ai = np.array([1.0] + list(ui * (theta ** np.arange(1, q + 1))))
            ar = np.array([1])
            ma_process = ArmaProcess(ar, ai)
            sim_data = ma_process.generate_sample(nsample=n)
            df = pd.DataFrame({
                'value': sim_data,
                'dataset': i,
                'theta': theta,
                'method': 'exponential'
            })
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def generate_powerlaw(q, gammas, n, num_datasets):
    all_data = []
    for gamma in gammas:
        for i in range(num_datasets):
            ai = np.array([1.0] + list(1 / (np.arange(1, q + 1) ** gamma)))
            ar = np.array([1])
            ma_process = ArmaProcess(ar, ai)
            sim_data = ma_process.generate_sample(nsample=n)
            df = pd.DataFrame({
                'value': sim_data,
                'dataset': i,
                'gamma': gamma,
                'method': 'powerlaw'
            })
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic MA time series data.")
    parser.add_argument("--q", type=int, required=True, help="Order of the MA process")
    parser.add_argument("--n", type=int, required=True, help="Length of each time series")
    parser.add_argument("--num_datasets", type=int, required=True, help="Number of datasets per parameter")
    parser.add_argument("--thetas", type=float, nargs='*', help="List of theta values for exponential mode")
    parser.add_argument("--gammas", type=float, nargs='*', help="List of gamma values for powerlaw mode")
    parser.add_argument("--output_exp", type=str, default="exp_ma.parquet", help="Output file for exponential MA")
    parser.add_argument("--output_power", type=str, default="powerlaw_ma.parquet", help="Output file for power-law MA")
    return parser.parse_args()

def main():
    args = parse_args()

    if not args.thetas and not args.gammas:
        raise ValueError("You must provide at least one of --thetas or --gammas.")

    if args.thetas:
        exp_df = generate_exp_decay_with_ui(q=args.q, thetas=args.thetas, n=args.n, num_datasets=args.num_datasets)
        exp_df.to_parquet(args.output_exp, index=False)
        print(f"Saved exponential MA dataset to {args.output_exp}")

    if args.gammas:
        power_df = generate_powerlaw(q=args.q, gammas=args.gammas, n=args.n, num_datasets=args.num_datasets)
        power_df.to_parquet(args.output_power, index=False)
        print(f"Saved power-law MA dataset to {args.output_power}")

if __name__ == "__main__":
    main()
