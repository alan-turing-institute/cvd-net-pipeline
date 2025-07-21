
import pandas as pd
import numpy as np
from utils.bayesian_calibration import BayesianCalibration
import os
from utils import plot_utils
import json
from datetime import datetime

def calibrate_parameters(data_type="synthetic",
                         n_samples:int=64, 
                         n_params:int=9, 
                         output_path:str='output', 
                         emulator_path:str=None,
                         output_keys:list=None,
                         include_timeseries:bool=True,
                         epsilon_obs_scale:float=0.05,
                         config:dict=None):

    if data_type == "synthetic":
    
        file_suffix = f'_{n_samples}_{n_params}_params'

        dir_name = f"{output_path}/output{file_suffix}"

        input_params = pd.read_csv(f'{output_path}/pure_input{file_suffix}.csv')

        # emulators
        emulators = pd.read_pickle(f"{dir_name}/emulators/linear_models_and_r2_scores_{n_samples}.pkl")
        
    elif data_type == "real":

        dir_name = output_path

        input_params = pd.read_csv(f'{emulator_path}/input_{n_samples}_{n_params}_params.csv')

        # emulators
        emulators = pd.read_pickle(f"{emulator_path}/output_{n_samples}_{n_params}_params/emulators/linear_models_and_r2_scores_{n_samples}.pkl")
        print(f"Using trained emulators from: {emulator_path}/output_{n_samples}_{n_params}_params.")

    # Data
    output_file = pd.read_csv(f"{dir_name}/waveform_resampled_all_pressure_traces_rv_with_pca.csv")

    # Direcotry for saving results
    output_dir = f"{dir_name}/bayesian_calibration_results/"

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

    if data_type == "synthetic":

        # Create the diagonal matrix
        e_obs = np.diag(diagonal_values) * epsilon_obs_scale
        
        bc = BayesianCalibration(input_prior=input_params, 
                                emulator_output=emulator_output, 
                                filtered_output=observation_data, 
                                which_obs=3, 
                                epsilon_obs = e_obs,
                                )

        bc.compute_posterior()

        # Save the posterior mean and covariance
        posterior_mean = pd.DataFrame(bc.Mu_post, index=bc.param_names, columns=['Posterior Mean'])    
        posterior_cov = pd.DataFrame(bc.Sigma_post, index=bc.param_names, columns=bc.param_names)

        
        # Sample from the posterior distribution
        bc.sample_posterior(n_samples=n_samples)

    elif data_type == "real":

        posterior_means = []

        # Create the diagonal matrix
        e_obs = np.diag(diagonal_values) * epsilon_obs_scale

        for row in range(len(observation_data)):
            bc = BayesianCalibration(input_prior=input_params, 
                                     emulator_output=emulator_output, 
                                     observation_data=observation_data.iloc[row:row+1], 
                                     epsilon_obs = e_obs,
                                     data_type=data_type)
            bc.compute_posterior()
            posterior_means.append(bc.Mu_post.squeeze())

        # Convert the list to a NumPy array
        posterior_means = np.array(posterior_means)
        Sigma_post = bc.Sigma_post

        # Save the posterior mean and covariance
        posterior_mean = pd.DataFrame(posterior_means, columns=bc.param_names)    
        posterior_cov = pd.DataFrame(Sigma_post, index=bc.param_names, columns=bc.param_names)

    n_output_keys =  len(all_output_keys)

    # Define the output directory name, appending the number of output keys to the directory name and including a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_bayesian = f"{output_dir}/{n_output_keys}_output_keys/calibration_{timestamp}"

    # Check if the directory exists, if not, create it
    if not os.path.exists(output_dir_bayesian):
        os.makedirs(output_dir_bayesian)

    posterior_mean.to_csv(f"{output_dir_bayesian}/posterior_mean.csv", index=False)
    posterior_cov.to_csv(f"{output_dir_bayesian}/posterior_covariance.csv", index=False)

    if data_type == "synthetic":
        bc.samples_df.to_csv(f"{output_dir_bayesian}/posterior_samples.csv", index=False)

    # Save the config file
    with open(os.path.join(output_dir_bayesian, 'used_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    if data_type == "synthetic":

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
        
    elif data_type == "real":

        plot_utils.plot_parameter_trajectories(Sigma_post=Sigma_post,
                                               posterior_means=posterior_means,
                                               bc=bc,
                                               output_path=output_dir_bayesian)

    
    return output_dir_bayesian, e_obs