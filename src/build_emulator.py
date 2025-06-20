import os
import pandas as pd
from utils import helper_functions


# steps/build_emulator.py
def build_emulator(n_samples:int=200, 
                   n_params:int=9, 
                   output_path:str="output", 
                   output_file_name:str="waveform_resampled_all_pressure_traces_rv_with_pca.csv"):
    
    file_sufix = f'_{n_samples}_{n_params}_params'

    input_file = pd.read_csv(f"{output_path}/input{file_sufix}.csv")
    output_file = pd.read_csv(f"{output_path}/output{file_sufix}/{output_file_name}")

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

    # Iterate through the output keys:

    # Here, we take each output, such as the raw pressure traces, calculated clinical indices and the PCA components,
    # (if present) and fit a linear model to each of them using those parameters that have been selected as relevant inputs.
    # Currently, just a linear regression model is used.
    for key in output_keys:
        model, r2, mse, rse = helper_functions.emulate_linear(input=filtered_input, output=output_file[key])
        linear_r2_scores[key] = r2
        linear_mse_scores[key] = mse
        linear_rse_scores[key] = rse
        fitted_models[key] = model

    # Convert the dictionaries to a DataFrame
    emulator_results_df = pd.DataFrame({'R2_Score': linear_r2_scores, 
                                        'MSE': linear_mse_scores,
                                        'RSE': linear_rse_scores, 
                                        'Model': fitted_models})
    
    # Create directory for output if it doesn't exist
    if not os.path.exists(f"{output_path}/output{file_sufix}/emulators"):
        os.makedirs(f"{output_path}/output{file_sufix}/emulators")

    # Save the DataFrame to a CSV file
    emulator_results_df.to_csv(f'{output_path}/output{file_sufix}/emulators/linear_models_and_r2_scores_{n_samples}.csv')

    # To save the DataFrame with models, use pickle
    emulator_results_df.to_pickle(f'{output_path}/output{file_sufix}/emulators/linear_models_and_r2_scores_{n_samples}.pkl')
   
    
