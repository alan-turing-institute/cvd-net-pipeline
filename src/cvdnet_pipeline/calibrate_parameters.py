import pandas as pd
import numpy as np
from cvdnet_pipeline.utils.bayesian_calibration import BayesianCalibration
import os
from cvdnet_pipeline.utils import plot_utils
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
                         dummy_data_dir:str=None,
                         config:dict=None):

    if data_type == "synthetic":
    
        file_suffix = f'_{n_samples}_{n_params}_params'
        dir_name = f"{output_path}/output{file_suffix}"

        input_params = pd.read_csv(f'{output_path}/pure_input{file_suffix}.csv')
        
        # True input parameters of dummy data
        true_input = pd.read_csv(f"{dummy_data_dir}/input_dummy_data.csv")

        # Synthetic dummy data (to calibrate on)
        output_file = pd.read_csv(f"{dummy_data_dir}/output_dummy_data/waveform_resampled_all_pressure_traces_rv_with_pca.csv")

        # Emulators
        emulators = pd.read_pickle(f"{dir_name}/emulators/linear_models_and_r2_scores_{n_samples}.pkl")
        
    elif data_type == "real":

        dir_name = output_path

        # Real data (to calibrate on)
        output_file = pd.read_csv(f"{dir_name}/waveform_resampled_all_pressure_traces_rv_with_pca.csv")

        input_params = pd.read_csv(f'{emulator_path}/pure_input_{n_samples}_{n_params}_params.csv')

        # Emulators
        emulators = pd.read_pickle(f"{emulator_path}/output_{n_samples}_{n_params}_params/emulators/linear_models_and_r2_scores_{n_samples}.pkl")
        print(f"Using trained emulators from: {emulator_path}/output_{n_samples}_{n_params}_params.")

    # Directory for saving results
    output_dir = f"{dir_name}/bayesian_calibration_results/"
    # Make directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # --- Build the observation noise --
    if include_timeseries:
        all_output_keys = output_file.iloc[:, :101].columns.tolist() + output_keys
        print("Including time-series in calibration as specified in config file.")

        # Build the diagonal entries: 101 ones followed by the variances
        # 101 ones are scaled by epsilon_obs_scale so they will equal 
        # 1 when multipled by epsilon_obs_scale further down. 
        var_values = output_file[output_keys].var().values
        diagonal_values = np.concatenate([np.ones(101)/epsilon_obs_scale, var_values]) 
    else:
        all_output_keys = output_keys
        var_values = output_file[output_keys].var().values
        diagonal_values = var_values

    # Select emulators and data for specified output_keys
    emulator_output = emulators.loc[all_output_keys]
    observation_data = output_file.loc[:, all_output_keys] 
    
    # --- Synthetic data calibration ---
    if data_type == "synthetic":

        # Create diagonal matrix of observation noise 
        e_obs = np.diag(diagonal_values) * epsilon_obs_scale
        
        bc = BayesianCalibration(input_prior=input_params, 
                                 emulator_output=emulator_output, 
                                 filtered_output=observation_data, 
                                 which_obs=3, 
                                 epsilon_obs=e_obs)

        bc.compute_posterior()

        # Save posterior mean and covariance
        posterior_mean = pd.DataFrame(bc.Mu_post, index=bc.param_names, columns=['Posterior Mean'])    
        posterior_cov = pd.DataFrame(bc.Sigma_post, index=bc.param_names, columns=bc.param_names)
        
        # Sample from the posterior
        bc.sample_posterior(n_samples=n_samples)

    # --- Real data calibration ---
    elif data_type == "real":

        posterior_means = []
        posterior_covariances = []

        # Create diagonal matrix of observation noise
        e_obs = np.diag(diagonal_values) * epsilon_obs_scale

        for row in range(len(observation_data)):
            bc = BayesianCalibration(
                input_prior=input_params, 
                emulator_output=emulator_output, 
                observation_data=observation_data.iloc[row:row+1], 
                epsilon_obs=e_obs,
                data_type=data_type
            )
            bc.compute_posterior()
            posterior_means.append(bc.Mu_post.squeeze())
            posterior_covariances.append(bc.Sigma_post)

        # Convert lists to arrays
        posterior_means = np.array(posterior_means)                  # (n_times, n_params)
        posterior_covariances = np.array(posterior_covariances)      # (n_times, n_params, n_params)

        # Convert means to DataFrame
        posterior_mean = pd.DataFrame(posterior_means, columns=bc.param_names)

    # --- Output directory and file saving ---
    n_output_keys = len(all_output_keys)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_bayesian = f"{output_dir}/{n_output_keys}_output_keys/calibration_{timestamp}"
    os.makedirs(output_dir_bayesian, exist_ok=True)

    # Save posterior means
    posterior_mean.to_csv(f"{output_dir_bayesian}/posterior_mean.csv", index=False)

    if data_type == "synthetic":
        posterior_cov.to_csv(f"{output_dir_bayesian}/posterior_covariance.csv", index=False)

    elif data_type == "real":
        # --- Save all covariance matrices ---
        np.save(f"{output_dir_bayesian}/posterior_covariances.npy", posterior_covariances)

        # --- Save diagonal variances (per time step) ---
        posterior_var = np.array([np.diag(Sigma_t) for Sigma_t in posterior_covariances])
        posterior_var_df = pd.DataFrame(posterior_var, columns=[f"{p}_var" for p in bc.param_names])
        posterior_var_df.to_csv(f"{output_dir_bayesian}/posterior_variances.csv", index=False)

    # --- Synthetic posterior samples ---
    if data_type == "synthetic":
        bc.samples_df.to_csv(f"{output_dir_bayesian}/posterior_samples.csv", index=False)
        
        # Remove negative samples
        cleaned_samples = bc.samples_df[(bc.samples_df >= 0).all(axis=1)]

         # Flag number of posterior samples that were removed during cleaning
        missing_indices = set(bc.samples_df.index) - set(cleaned_samples.index)
        if missing_indices:
            print(f"The following posterior_samples were removed due to negativity: {sorted(missing_indices)}")
            pd.Series(sorted(missing_indices)).to_csv(f"{output_dir_bayesian}/cleaned_posterior_samples_indices.csv", index=False)
       
        cleaned_samples.to_csv(f"{output_dir_bayesian}/cleaned_posterior_samples.csv", index=False)

    # --- Save config file ---
    with open(os.path.join(output_dir_bayesian, 'used_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # --- Plotting ---
    if data_type == "synthetic":
        plot_utils.plot_posterior_distributions(true_input, 
                                                bc.mu_0,
                                                bc.Sigma_0,
                                                bc.Mu_post,
                                                bc.Sigma_post,
                                                bc.which_obs,
                                                bc.param_names,
                                                output_path=output_dir_bayesian)

        plot_utils.plot_posterior_covariance_matrix(bc.Sigma_0,
                                                    bc.Sigma_post,
                                                    bc.param_names,
                                                    output_path=output_dir_bayesian)
        
    elif data_type == "real":
        plot_utils.plot_parameter_trajectories(Sigma_post=posterior_covariances,
                                               posterior_means=posterior_means,
                                               bc=bc,
                                               output_path=output_dir_bayesian)

    return output_dir_bayesian, e_obs