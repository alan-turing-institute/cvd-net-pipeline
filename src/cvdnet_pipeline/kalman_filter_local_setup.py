from datetime import datetime
import numpy as np
import pandas as pd
from cvdnet_pipeline.utils.kf_emulator_local import KalmanFilterWithLocalEmulator
from cvdnet_pipeline.utils.plot_utils import plot_kf_estimates
import os
import pickle

def KF_local_setup(n_samples:int=4096, 
                n_params:int=9, 
                emulator_path:str='emulator',
                output_path:str='output_synthetic', 
                output_keys:list=None,
                include_timeseries:bool=True,
                epsilon_obs_scale:float=0.05,
                data_type:str=None
                ):
        
    if data_type == 'synthetic':
        print("Using KF for synthetic data.")
        dir_output_name = f"{output_path}/output_{n_samples}_{n_params}_params"
        output_file = pd.read_csv(f"{dir_output_name}/waveform_resampled_all_pressure_traces_rv_with_pca.csv")

        # Input parameter samples to be used to build local emulators
        input_prior = pd.read_csv(f'{output_path}/input_{n_samples}_{n_params}_params.csv')
        input_prior_pure = pd.read_csv(f'{output_path}/pure_input_{n_samples}_{n_params}_params.csv')

    elif data_type == 'real':
        # Load real observation data
        output_file = pd.read_csv(f"{output_path}/waveform_resampled_all_pressure_traces_rv_with_pca.csv")

        # Input parameter samples to be used to build local emulators
        input_prior = pd.read_csv(f'{emulator_path}/input_{n_samples}_{n_params}_params.csv')
        input_prior_pure = pd.read_csv(f'{emulator_path}/pure_input_{n_samples}_{n_params}_params.csv')
    
    
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
    
    # Select observaton data to calibrate on
    observation_data = output_file.loc[:, all_output_keys]

    # Create the diagonal matrix of observation noise (for specified output keys)
    R = np.diag(diagonal_values) * epsilon_obs_scale

    ## Create initial prior for Kalman Filter
    mu_0 = np.array(input_prior.mean().loc[:'T'])
    mu_0 = mu_0.reshape(-1, 1)
    Sigma_0 = np.diag(input_prior.var().loc[:'T'])

    # dynamically define prior on T
    mu_0[-1,-1] = observation_data['iT'].iloc[0]
    Sigma_0[-1, -1] = 0.0001

    # Parameter names
    param_names = input_prior.loc[:, :'T'].columns.to_list()


    # Process noise covariance
    variances = input_prior.var().loc[:'T'].values
    Q = np.diag(0.01 * variances)
    
    # Give KF prior samples to build local emulators from
    if data_type == 'synthetic':
        Y_prior = pd.read_csv(f"{output_path}/output_{n_samples}_{n_params}_params/waveform_resampled_all_pressure_traces_rv_with_pca.csv")
        Y_prior = Y_prior.loc[:, all_output_keys]
    elif data_type == 'real':
        # simulated outputs for prior samples
        Y_prior = pd.read_csv(f"{emulator_path}/output_{n_samples}_{n_params}_params/waveform_resampled_all_pressure_traces_rv_with_pca.csv")
        Y_prior = Y_prior.loc[:, all_output_keys]

 

    # Initialize the Kalman Filter with Emulator
    kf = KalmanFilterWithLocalEmulator(X_prior=input_prior_pure, 
                                       Y_prior=Y_prior,
                                       observation_data=observation_data, 
                                       Q=Q, 
                                       R=R, 
                                       mu_0=mu_0, 
                                       Sigma_0=Sigma_0,
                                       k_neighbors=200)      

    # Run the filter
    estimates = kf.run(np.array(observation_data))

    # Save the resulting estimates

    # Define the output directory name, appending the number of output keys to the directory name and including a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if data_type == 'synthetic':
        dir_name = f"{dir_output_name}/kf_local_calibration_results/{len(all_output_keys)}_output_keys"
        os.makedirs(dir_name, exist_ok=True)
    elif data_type == 'real':
        dir_name = f"{output_path}/kf_local_calibration_results/{len(all_output_keys)}_output_keys"
        os.makedirs(dir_name, exist_ok=True)

    output_dir_kf = f"{dir_name}/calibration_{timestamp}"
    os.makedirs(output_dir_kf, exist_ok=True)

    # Save the estimated parameters to a CSV and npy files. First, turn the mu entries into a DataFrame
    mu_estimates_df = pd.DataFrame(
        np.array([estimate[0] for estimate in estimates]), 
        columns=param_names
    )
    sigma_estimates = np.array([estimate[1] for estimate in estimates])

    # Save to files
    mu_estimates_df.to_csv(f"{output_dir_kf}/kf_estimated_means.csv", index=False)
    np.save(f"{output_dir_kf}/kf_estimated_covariances.npy", sigma_estimates)

    # Save the entire estimates list as a pickle file
    with open(f"{output_dir_kf}/kf_estimated_means_and_covariances.pkl", 'wb') as f:
        pickle.dump(estimates, f)

    # Plot the results
    plot_kf_estimates(estimates=estimates, 
                      param_names=param_names,
                      output_path=output_dir_kf)
    
    # Save Q matrix and input prior variance
    Q_df = pd.DataFrame(Q, index=param_names, columns=param_names)
    Q_df.to_csv(f"{output_dir_kf}/process_noise_covariance_Q.csv")

    # (Optional) also save variances
    pd.DataFrame({"variance": variances}, index=param_names).to_csv(f"{output_dir_kf}/param_variances.csv")
    
    return estimates