import argparse
import numpy as np
import pandas as pd

def generate_divisible_number(d_model: np.ndarray, nhead: np.ndarray, num_runs: int) -> np.ndarray:
    """
    Generate random numbers for d_model that are divisible by corresponding nhead values.
    
    Args:
        d_model: Array of initial d_model values
        nhead: Array of nhead values (must be same length as d_model)
        num_runs: Number of values to generate (length of arrays)
        
    Returns:
        Array of d_model values where each d_model[i] is divisible by nhead[i]
    """
    
    # Generate divisible numbers
    for i in range(num_runs):
        while d_model[i] % nhead[i] != 0:
            d_model[i] = np.random.randint(1, 11)
    
    return d_model


def generate_patchtst_hyperparameters(num_runs: int, seed: int) -> pd.DataFrame:
    """
    Generate random hyperparameters for PatchTST model experiments.
    
    Args:
        num_runs: Number of hyperparameter sets to generate
        max_epochs: Maximum number of training epochs (currently unused)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame containing randomly generated hyperparameters with columns:
        - context_length: Random int between 1-100
        - stride: Random int between 1-20
        - d_model: Random int between 1-32 (divisible by nhead)
        - nhead: Random int between 1-10
        - num_encoder_layers: Random int between 1-5
    """
    np.random.seed(seed)
    hyperparameters_dict = {
        "context_length": np.random.randint(1, 100, num_runs),
        "stride": np.random.randint(1, 20, num_runs),
        "d_model": np.random.randint(1, 32, num_runs),
        "nhead": np.random.randint(1, 10, num_runs),
        "num_encoder_layers":np.random.randint(1, 5, num_runs),

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

    # Ensure d_model is divisible by nhead
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
    parser.add_argument("--seed", type=int, required=True, help="seed")

    cmd_args = parser.parse_args()
    num_runs = cmd_args.num_runs
    seed = cmd_args.seed

    ################################################################################
    # Run the generator
    ################################################################################
    df = generate_patchtst_hyperparameters(num_runs, seed)

    ################################################################################
    # Save the data
    ################################################################################
    df.to_parquet(f'../data/PatchTST_hyperparams.parquet', index=False)
    print(f"Shape of the hyperparameter dataframe: {df.shape}")
    print(f"PatchTST hyperparameters' generation complete. Data is saved in ../data")