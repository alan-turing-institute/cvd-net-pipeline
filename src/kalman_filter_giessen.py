from datetime import datetime
import numpy as np
import pandas as pd
from utils.kf_emulator import KalmanFilterWithEmulator
from utils.plot_utils import plot_kf_estimates
import os

def KFGiessenSETUP(n_samples:int=4096, 
                n_params:int=9, 
                output_path:str='output', 
                emulator_path:str=None,
                output_keys:list=None,
                include_timeseries:bool=True,
                epsilon_obs_scale:float=0.05,
                ):
        
    # Load observation data
    output_file = pd.read_csv(f"{output_path}/waveform_resampled_all_pressure_traces_rv_with_pca.csv")

    # Input for priors
    input_prior = pd.read_csv(f'{emulator_path}/input_{n_samples}_{n_params}_params.csv')

    # emulators
    emulators = pd.read_pickle(f"{emulator_path}/output_{n_samples}_{n_params}_params/emulators/linear_models_and_r2_scores_{n_samples}.pkl")

    if include_timeseries:
        all_output_keys = output_file.iloc[:, :101].columns.tolist() + output_keys
        print("Including time-series in calibraiton as specified in config file.")

        # Build the diagonal entries: 101 ones followed by the std devs
        # 101 ones are scaled by epsilon_obs_scale so they will equal 
        # 1 when multipled by epsilon_obs_scale further down. 
        sd_values = output_file[output_keys].std().values
        diagonal_values = np.concatenate([np.ones(101)/epsilon_obs_scale, sd_values]) 
    else:
        all_output_keys = output_keys
        sd_values = output_file[output_keys].std().values
        diagonal_values = sd_values

    # Create the diagonal matrix
    e_obs = np.diag(diagonal_values) * epsilon_obs_scale
    
    # Select emulators and data for specified output_keys
    emulator_output = emulators.loc[all_output_keys]
    observation_data = output_file.loc[:, all_output_keys]

    # Priors
    mu_0 = np.array(input_prior.mean().loc[:'T'])
    mu_0 = mu_0.reshape(-1, 1)
    Sigma_0 = np.diag(input_prior.var().loc[:'T'])

    # dynamically define prior on T
    mu_0[-1,-1] = observation_data['iT'].iloc[0]
    Sigma_0[-1, -1] = 0.0000001

    # Parameter names
    param_names = input_prior.loc[:, :'T'].columns.to_list()

    # Model error
    epsilon_model = np.diag(emulator_output['RSE']**2) 

    # Construct beta matrix and intercepts
    beta_matrix = []
    intercept = []

    for _, row_entry in emulator_output.iterrows():
        model = row_entry['Model']
        beta_matrix.append(model.coef_)
        intercept.append(model.intercept_)
    
    beta_matrix = np.array(beta_matrix)
    intercept = np.array(intercept).reshape(len(intercept), 1)
    
    # Process noise covariance
    Q = np.eye(n_params) * 0.01

    # Initialize the Kalman Filter with Emulator
    kf = KalmanFilterWithEmulator(beta_matrix, 
                                  intercept.flatten(), 
                                  Q, 
                                  e_obs, 
                                  epsilon_model, 
                                  mu_0.flatten(), 
                                  Sigma_0)

    # Run the filter
    estimates = kf.run(np.array(observation_data))

    # Save the resulting estimates

    # Define the output directory name, appending the number of output keys to the directory name and including a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_kf = f"{output_path}/kf_results/{len(all_output_keys)}_output_keys/calibration_{timestamp}"
    os.makedirs(output_dir_kf, exist_ok=True)

    # Save the estimated parameters to a CSV file. First, turn the mu entries into a DataFrame
    mu_estimates_df = pd.DataFrame(
        np.array([estimate[0] for estimate in estimates]), 
        columns=param_names
    )
    mu_estimates_df.to_csv(f"{output_dir_kf}/kf_estimated_parameters.csv", index=False)

    # Save the whole estimates (mean and covariance) to a npy file
    np.save(f"{output_dir_kf}/kf_estimates.npy", estimates)

    # Plot the results
    plot_kf_estimates(estimates=estimates, 
                      param_names=param_names,
                      output_path=output_dir_kf)

    return estimates