import argparse
import numpy as np
import pandas as pd
import random

def generate_divisible_number(d_model: np.ndarray, nhead: np.ndarray, num_runs: int) -> np.ndarray:
    """Ensure d_model is divisible by nhead"""
    
    # Generate divisible numbers
    for i in range(num_runs):
        while d_model[i] % nhead[i] != 0:
            d_model[i] = random.randint(1, 10)
    
    return d_model


def generate_patchtst_hyperparameters(num_runs: int, max_epochs: int, seed: int) -> pd.DataFrame:
    np.random.seed(seed)
    hyperparameters_dict = {
        "context_length": np.random.randint(1, 100, num_runs),
        "stride": np.random.randint(1, 20, num_runs),
        "d_model": np.random.randint(1, 32, num_runs),
        "nhead": np.random.randint(1, 10, num_runs),
        "num_encoder_layers":np.random.randint(1, 5, num_runs),

        # "max_epochs": max_epochs, 

        # Static
        # "patch_len": 1,  # Static value for AR(1) 
        # "scaling": "mean", 
        # "activation": "relu", 
        # "dim_feedforward": 128, 
        # "dropout": 0.1,  
        # "norm_first": False, 
        # "padding_patch": "end", 
        # "num_feat_dynamic_real": 0,  
    }

    # Convert the dictionary to a DataFrame
    df_hyperparam = pd.DataFrame(hyperparameters_dict)

    d_model = df_hyperparam["d_model"].values
    nhead = df_hyperparam["nhead"].values
    df_hyperparam["d_model"] = generate_divisible_number(d_model, nhead, num_runs)                                                    
    
    return df_hyperparam

if __name__ == "__main__":
    print("Generate hyperparameters for PatchTST Model")

    ################################################################################
    # Parse the command-line arguments
    ################################################################################

    parser = argparse.ArgumentParser(description="Generate hyperparameters for PatchTST Model")
    parser.add_argument("--num_runs", type=int, required=True, help="Number of models")
    parser.add_argument("--max_epochs", type=int, required=True, help="Maximun of epochs for training")
    parser.add_argument("--seed", type=int, required=True, help="seed")

    cmd_args = parser.parse_args()
    num_runs = cmd_args.num_runs
    max_epochs = cmd_args.max_epochs
    seed = cmd_args.seed

    ################################################################################
    # Run the generator
    ################################################################################
    df = generate_patchtst_hyperparameters(num_runs, max_epochs, seed)

    ################################################################################
    # Save the data
    ################################################################################
    df.to_parquet(f'../data/PatchTST_hyperparams.parquet', index=False)
    print(f"Shape of the hyperparameter dataframe: {df.shape}")
    print(f"PatchTST hyperparameters' generation complete. Data is saved in ../data")