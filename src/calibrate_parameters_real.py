
import pandas as pd
import numpy as np
from utils.bayesian_calibration_real import BayesianCalibrationGiessen
import os
from utils import plot_utils
import json
from datetime import datetime
import matplotlib.pyplot as plt

def calibrate_parameters_real(n_samples:int=50, 
                         n_params:int=9, 
                         output_path:str='output', 
                         emulator_path:str=None,
                         output_keys:list=None,
                         include_timeseries:bool=True,
                         epsilon_obs_scale:float=0.05,
                         config:dict=None):


    # Data
    output_file = pd.read_csv(f"{output_path}/waveform_resampled_all_pressure_traces_rv_with_pca.csv")
    
    # Input for priors
    input_params = pd.read_csv(f'{emulator_path}/input_{n_samples}_{n_params}_params.csv')

    # emulators
    emulators = pd.read_pickle(f"{emulator_path}/output_{n_samples}_{n_params}_params/emulators/linear_models_and_r2_scores_{n_samples}.pkl")
    print(f"Using trained emulators from: {emulator_path}/output_{n_samples}_{n_params}_params.")

    # Direcotry for saving results
    output_dir = f"{output_path}/bayesian_calibration_results/"

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


    posterior_means = []


    # Create the diagonal matrix
    e_obs = np.diag(diagonal_values) * epsilon_obs_scale

    for row in range(len(observation_data)):
        bc = BayesianCalibrationGiessen(input_params, emulator_output, observation_data.iloc[row:row+1], epsilon_obs = e_obs)
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

    # Save the config file
    with open(os.path.join(output_dir_bayesian, 'used_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
 
    class ResolutionController:
        def __init__(self, window_size):
            self.window_size = window_size

        def downsample(self, data):
            """Downsamples the data by averaging over non-overlapping windows."""
            if data.shape[0] < self.window_size:
                raise ValueError(f"Data has fewer than {self.window_size} time steps!")

            num_windows = data.shape[0] // self.window_size  # Compute number of full windows
            return data[:num_windows * self.window_size].reshape(num_windows, self.window_size, -1).mean(axis=1)

    # Initialize resolution controller
    window_size = 5
    res_controller = ResolutionController(window_size)

    # Define time range before downsampling
    time_range = (1, 4000)  # Specify the indices from the original data

    # Ensure posterior_variances has shape (3888, p)
    posterior_variances_corrected = np.array(Sigma_post).diagonal().reshape(1, -1)  # (1, p)
    posterior_variances_corrected = np.tile(posterior_variances_corrected, (posterior_means.shape[0], 1))  # (3888, p)


    # Slice the original data before downsampling
    posterior_means_trimmed = posterior_means[time_range[0]:time_range[1]]
    posterior_variances_trimmed = posterior_variances_corrected[time_range[0]:time_range[1]]

    # Downsample the sliced data
    posterior_means_smooth = res_controller.downsample(posterior_means_trimmed)  # (new_length, p)
    posterior_variances_smooth = res_controller.downsample(np.sqrt(posterior_variances_trimmed))  # (new_length, p)


    # Generate new time indices based on downsampling
    T_smooth = np.arange(posterior_means_smooth.shape[0]) * window_size + time_range[0]

    # Colors for different parameters
    param_names = bc.param_names
    colors = plt.cm.get_cmap('Set1', len(param_names)).colors

    # Plot each parameter on a separate subplot
    fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 8), sharex=True)

    for i in range(len(param_names)):
        mean = posterior_means_smooth[:, i]  # Smoothed mean
        std_dev = posterior_variances_smooth[:, i]  # Smoothed standard deviation

        axes[i].plot(T_smooth, mean, color=colors[i], label=param_names[i])
        axes[i].fill_between(T_smooth, mean - 2 * std_dev, mean + 2 * std_dev, color=colors[i], alpha=0.2)

        axes[i].set_ylabel('Value')
        axes[i].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        axes[i].grid()

    axes[-1].set_xlabel('Time')
    fig.suptitle(f'Parameter Trajectories (Averaged Over {window_size} Steps) [Original Range: {time_range}]')

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make space for legends on the right
    plt.show()
    
    return 