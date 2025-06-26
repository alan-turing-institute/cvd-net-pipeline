
import pandas as pd
import numpy as np
from utils.bayesian_calibration_combined import BayesianCalibration
import os
from utils import plot_utils
import json
from datetime import datetime

def calibrate_parameters(n_samples:int=50, 
                         n_params:int=9, 
                         output_path:str='output', 
                         output_keys:list=None,
                         include_timeseries:bool=True,
                         epsilon_obs_scale:float=0.05,
                         config:dict=None):


    file_suffix = f'_{n_samples}_{n_params}_params'

    # Data
    input_params = pd.read_csv(f'{output_path}/input{file_suffix}.csv')
    output_file = pd.read_csv(f"{output_path}/output{file_suffix}/waveform_resampled_all_pressure_traces_rv_with_pca.csv")

    # emulators
    emulators = pd.read_pickle(f"{output_path}/output{file_suffix}/emulators/linear_models_and_r2_scores_{n_samples}.pkl")
    

    # Direcotry for saving results
    output_dir = f"{output_path}/output{file_suffix}/bayesian_calibration_results/"

    # Make directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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


    # Select emulators and data for specified output_keys
    emulator_output = emulators.loc[all_output_keys]
    observation_data = output_file.loc[:, all_output_keys]
        
    # Create the diagonal matrix
    e_obs = np.diag(diagonal_values) * epsilon_obs_scale
    
    bc = BayesianCalibration(input_params, emulator_output, observation_data, which_obs=3, epsilon_obs = e_obs)

    bc.compute_posterior()

    # Save the posterior mean and covariance
    posterior_mean = pd.DataFrame(bc.Mu_post, index=bc.param_names, columns=['Posterior Mean'])    
    posterior_cov = pd.DataFrame(bc.Sigma_post, index=bc.param_names, columns=bc.param_names)

    
    # Smaple from the posterior distribution
    bc.sample_posterior(n_samples=n_samples)

    n_output_keys =  len(all_output_keys)

    # Define the output directory name, appending the number of output keys to the directory name and including a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_bayesian = f"{output_dir}/{n_output_keys}_output_keys/calibration_{timestamp}"

    # Check if the directory exists, if not, create it
    if not os.path.exists(output_dir_bayesian):
        os.makedirs(output_dir_bayesian)

    bc.samples_df.to_csv(f"{output_dir_bayesian}/posterior_samples.csv", index=False)
    posterior_mean.to_csv(f"{output_dir_bayesian}/posterior_mean.csv", index=False)
    posterior_cov.to_csv(f"{output_dir_bayesian}/posterior_covariance.csv", index=False)

    # Save the config file
    with open(os.path.join(output_dir_bayesian, 'used_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
   
    # Plot the prior and posteior distributions
    plot_utils.plot_posterior_distributions(input_params, 
                                             bc.mu_0,
                                             bc.Sigma_0,
                                             bc.Mu_post,
                                             bc.Sigma_post,
                                             bc.which_obs,
                                             bc.param_names,
                                             output_path=output_dir_bayesian)

    # Plot posterior covariance matrix
    plot_utils.plot_posterior_covariance_matrix(bc.Sigma_0,
                                                 bc.Sigma_post,
                                                 bc.param_names,
                                                 output_path=output_dir_bayesian)

    
    return output_dir_bayesian, e_obs