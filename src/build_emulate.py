import contextlib
import sklearn
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
def build_emulate(n_samples:int=500, n_params:int=5, n_pca_components:int=10, output_path:str="output"):
    print("[BuildEmulator] training emulator (placeholder)")

    
    input_file = pd.read_csv(f"{output_path}/input_{n_samples}_{n_params}params.csv")
    output_file = pd.read_csv(f"{output_path}/output_{n_samples}_{n_params}params/resampled_all_pressure_traces_rv.csv")

    utils.emulate_linear(input_file, output=output_file['iT'])

    # Initialize dictionaries to store R2 scores and models
    linear_r2_scores = {}
    linear_mse_scores = {}
    linear_rse_scores = {}
    fitted_models = {}

    # List of output keys to process
    #output_keys = ['CO', 'EF', 'mPAP', 'dPAP', 'sPAP', 'PC1', 'PC2', 'PC3', 'PC4']
    output_keys = output_file.iloc[:,101:].columns
    #output_keys = ['t_max_dpdt', 'a_epad', 'epad', 's_a_epad', 's_epad', 'min_dpdt', 'max_dpdt',
    #               'A_p', 'P_max', 'esp', 'sys', 'EF',  'Ees/Ea', 'PC1', 'PC2', 'PC3']

    # Iterate through the output keys
    for key in output_keys:
        model, r2, mse, rse = utils.emulate_linear(input=input_file, output=output_file[key])
        linear_r2_scores[key] = r2
        linear_mse_scores[key] = mse
        linear_rse_scores[key] = rse
        fitted_models[key] = model

    # Convert the dictionaries to a DataFrame
    results_df = pd.DataFrame({'R2_Score': linear_r2_scores, 'MSE': linear_mse_scores,  'RSE': linear_rse_scores, 'Model': fitted_models})
    # Now `results_df` will be a DataFrame with column names as indices, R2 scores, and models
    print(results_df)

    # Save the DataFrame to a CSV file (models will not be saved in this step)
    results_df.to_csv(f'../Emulation/Outputs/Output_{n_samples}_{n_params}params/Emulators/linear_models_and_r2_scores_{n_samples}.csv')

    # To save the DataFrame with models, use pickle
    results_df.to_pickle(f'../Emulation/Outputs/Output_{n_samples}_{n_params}params/Emulators/linear_models_and_r2_scores_{n_samples}.csv')
    results_df.to_pickle(f'/Users/pmzff/Documents/GitHub/GiessenDataAnalysis/Emulators/linear_models_and_r2_scores_{n_samples}.csv')

emulate()