from datetime import datetime
import numpy as np
import pandas as pd
from cvdnet_pipeline.utils.kf_emulator import KalmanFilterWithEmulator
from cvdnet_pipeline.utils.plot_utils import plot_kf_estimates
import os
import pickle

def KFGiessenSETUP(n_samples:int=4096, 
                n_params:int=9, 
                output_path:str='output', 
                emulator_path:str=None,
                output_keys:list=None,
                include_timeseries:bool=True,
                epsilon_obs_scale:float=0.05,
                data_type:str=None
                ):
        
    if data_type == 'synthetic':
        print("Using KF for synthetic data.")
        dir_output_name = f"{output_path}/output_{n_samples}_{n_params}_params"
        output_file = pd.read_csv(f"{dir_output_name}/waveform_resampled_all_pressure_traces_rv_with_pca.csv")
    elif data_type == 'real':
        # Load observation data
        output_file = pd.read_csv(f"{output_path}/waveform_resampled_all_pressure_traces_rv_with_pca.csv")

    # Input for priors
    input_prior = pd.read_csv(f'{emulator_path}/input_{n_samples}_{n_params}_params.csv')
    
    # emulators
    emulators = pd.read_pickle(f"{emulator_path}/output_{n_samples}_{n_params}_params/emulators/linear_models_and_r2_scores_{n_samples}.pkl")

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
    Sigma_0[-1, -1] = 0.0001

    # Parameter names
    param_names = input_prior.loc[:, :'T'].columns.to_list()

    # Model error
    epsilon_model = np.diag(emulator_output['MSE']) 

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
    variances = input_prior.var().loc[:'T'].values
    means = input_prior.mean().loc[:'T'].values
    Q = np.diag(0.01 * variances)
    

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

    if data_type == 'synthetic':
        dir_name = f"{dir_output_name}/kf_calibration_results/{len(all_output_keys)}_output_keys"
        os.makedirs(dir_name, exist_ok=True)
    elif data_type == 'real':
        dir_name = f"{output_path}/kf_calibration_results/{len(all_output_keys)}_output_keys"
        os.makedirs(dir_name, exist_ok=True)

    output_dir_kf = f"{dir_name}/kf_calibration_results/{len(all_output_keys)}_output_keys/calibration_{timestamp}"
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