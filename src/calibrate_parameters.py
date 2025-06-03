
import pandas as pd
import numpy as np
from utils.bayesian_calibration import BayesianCalibration
import os
from utils import plot_utils

def calibrate_parameters(n_samples:int=50, 
                         n_params:int=9, 
                         output_path:str='output', 
                         output_keys:list=None,
                         config:dict=None):


    # Data
    input_params = pd.read_csv(f'{output_path}/input_{n_samples}_{n_params}params.csv')
    output_file = pd.read_csv(f"{output_path}/output_{n_samples}_{n_params}params/waveform_resampled_all_pressure_traces_rv_with_pca.csv")

    # emulators
    emulators = pd.read_pickle(f"{output_path}/output_{n_samples}_{n_params}params/emulators/linear_models_and_r2_scores_{n_samples}.pkl")
    

    # Direcotry for saving results
    output_dir = f"{output_path}/output_{n_samples}_{n_params}params/bayesian_calibration_results/"

    # Make directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    emulator_output = emulators.loc[output_keys]
    filtered_output = output_file.loc[:, output_keys]

    bc = BayesianCalibration(input_params, emulator_output, filtered_output, which_obs=3, epsilon_obs_scale=0.05)

    bc.compute_posterior()

    # Save the posterior mean and covariance
    posterior_mean = pd.DataFrame(bc.Mu_post, index=bc.param_names, columns=['Posterior Mean'])    
    posterior_cov = pd.DataFrame(bc.Sigma_post, index=bc.param_names, columns=bc.param_names)

    
    # Smaple from the posterior distribution
    bc.sample_posterior(n_samples=1000)

    n_output_keys =  len(output_keys)

    # Define the output directory name
    output_dir_bayesian = f"{output_dir}/bayesian_calibration_results/{n_output_keys}_output_keys"

    # Check if the directory exists
    if os.path.exists(output_dir_bayesian):
        # Load the output keys from the existing directory and compare to the current output keys


        # If the output keys are the same, overwrite the existing files



        # If the output keys are different, create a new directory with a different name


    bc.samples_df.to_csv(f"{output_dir_bayesian}/posterior_samples.csv", index=False)
    posterior_mean.to_csv(f"{output_dir_bayesian}/posterior_mean.csv")
    posterior_cov.to_csv(f"{output_dir_bayesian}/posterior_covariance.csv")

   
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

    
    