import contextlib
import sklearn
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pandas as pd
from utils import utils, plot_utils


# steps/build_emulator.py
def build_emulator(n_samples:int=500, n_params:int=5, output_path:str="output", output_file_name:str="resampled_all_pressure_traces_rv_with_pca.csv"):
    print("[BuildEmulator] training emulator (placeholder)")

    
    input_file = pd.read_csv(f"{output_path}/input_{n_samples}_{n_params}params.csv")
    output_file = pd.read_csv(f"{output_path}/output_{n_samples}_{n_params}params/{output_file_name}")


    # Select relevant inputs only
    relevant_columns = []
    for col in input_file.columns:
        relevant_columns.append(col)
        if col == 'T': break


    # Select only first relevant inputs 
    filtered_input = input_file[relevant_columns]

    # List of output keys to process
    output_keys = output_file.columns
    
    # Initialize dictionaries to store R2 scores and models
    linear_r2_scores = {}
    linear_mse_scores = {}
    linear_rse_scores = {}
    fitted_models = {}



    # Iterate through the output keys
    for key in output_keys:
        model, r2, mse, rse = utils.emulate_linear(input=filtered_input, output=output_file[key])
        linear_r2_scores[key] = r2
        linear_mse_scores[key] = mse
        linear_rse_scores[key] = rse
        fitted_models[key] = model

    # Convert the dictionaries to a DataFrame
    emulator_results_df = pd.DataFrame({'R2_Score': linear_r2_scores, 'MSE': linear_mse_scores,  'RSE': linear_rse_scores, 'Model': fitted_models})
    
    # Create directory for output if it doesn't exist
    if not os.path.exists(f"{output_path}/output_{n_samples}_{n_params}params/emulators"):
        os.makedirs(f"{output_path}/output_{n_samples}_{n_params}params/emulators")

    # Save the DataFrame to a CSV file
    emulator_results_df.to_csv(f'{output_path}/output_{n_samples}_{n_params}params/emulators/linear_models_and_r2_scores_{n_samples}.csv')

    # To save the DataFrame with models, use pickle
    emulator_results_df.to_pickle(f'{output_path}/output_{n_samples}_{n_params}params/emulators/linear_models_and_r2_scores_{n_samples}.pkl')
   
    
