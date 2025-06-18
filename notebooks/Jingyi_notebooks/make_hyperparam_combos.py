import argparse
import numpy as np
import pandas as pd

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
    rng = np.random.default_rng(seed)
    hyperparameters_dict = {
        "context_length": rng.integers(1, 100, num_runs),
        "stride": rng.integers(1, 20, num_runs),
        "nhead": rng.integers(1, 10, num_runs),
        "num_encoder_layers": rng.integers(1, 5, num_runs),

        # Static
        # "patch_len": 1,  # Static value for AR(1) 
        # "scaling": "mean", 
        # "activation": "relu", 
        # "dim_feedforward": 128, 
        # "dropout": 0.1,  
        # "norm_first": False, 
        # "padding_patch": "end", 
        # "num_feat_dynamic_real": 0,  

            # Parameters:
            # context_length (int, default = 96) – Number of time units that condition the predictions

            # patch_len (int, default = 16) – Length of the patch.

            # stride (int, default = 8) – Stride of the patch.

            # d_model (int, default = 32) – Size of hidden layers in the Transformer encoder.

            # nhead (int, default = 4) – Number of attention heads in the Transformer encoder which must divide d_model.

            # num_encoder_layers (int, default = 2) – Number of layers in the Transformer encoder.

            # distr_output (gluonts.torch.distributions.Output, default = StudentTOutput()) – Distribution output object that defines how the model output is converted to a forecast, and how the loss is computed.

            # scaling ({"mean", "std", None}, default = "mean") –

            # Scaling applied to each context window during training & prediction. One of "mean" (mean absolute scaling), "std" (standardization), None (no scaling).

            # Note that this is different from the target_scaler that is applied to the entire time series.

            # max_epochs (int, default = 100) – Number of epochs the model will be trained for

            # batch_size (int, default = 64) – Size of batches used during training

            # num_batches_per_epoch (int, default = 50) – Number of batches processed every epoch

            # lr (float, default = 1e-3,) – Learning rate used during training

            # weight_decay (float, default = 1e-8) – Weight decay regularization parameter.

            # keep_lightning_logs (bool, default = False) – If True, lightning_logs directory will NOT be removed after the model finished training.

            # target_scaler ({“standard”, “mean_abs”, “robust”, “min_max”, None}, default = None) 

            # covariate_scaler ({“global”, None})
    }

    # Convert the dictionary to a DataFrame
    df_hyperparam = pd.DataFrame(hyperparameters_dict)

    df_hyperparam["d_model"] = rng.integers(1, 32 // df_hyperparam["nhead"], num_runs) * \
                                df_hyperparam["nhead"]
   
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