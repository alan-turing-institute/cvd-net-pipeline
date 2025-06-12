import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd
import math

class BayesianCalibrationGiessen:
    def __init__(self, input_prior, emulator_output, observation_data, 
                 epsilon_obs_scale, epsilon_alt=None):
        self.input_prior = input_prior
        self.emulator_output = emulator_output
        self.observation_data = observation_data
        self.epsilon_alt = epsilon_alt 

        # Priors
        self.mu_0 = np.array(input_prior.mean().loc[:'T'])
        self.mu_0 = self.mu_0.reshape(-1, 1)
        self.Sigma_0 = np.diag(input_prior.var().loc[:'T'])

        # dynamically define prior on T
        self.mu_0[-1,-1] = observation_data['iT'].iloc[0]
        self.Sigma_0[-1, -1] = 0.0000001
        
        # Parameter names
        self.param_names = input_prior.loc[:, :'T'].columns.to_list()
        

        # Model error
        self.epsilon_model = np.diag(emulator_output['RSE']**2) 
       
        
        # Observation error
        self.obs_error_scale = epsilon_obs_scale 
        default_epsilon_obs = np.diag(observation_data.std(axis=0) * self.obs_error_scale)  
        self.epsilon_obs = default_epsilon_obs if epsilon_alt is None else self.epsilon_alt*self.obs_error_scale
        
        # Compute posterior
        self.compute_posterior()
        

 
    def compute_posterior(self):
        full_error = self.epsilon_obs + self.epsilon_model
       
        # Construct beta matrix and intercepts
        beta_matrix = []
        intercept = []

        for _, row_entry in self.emulator_output.iterrows():
            model = row_entry['Model']
            beta_matrix.append(model.coef_)
            intercept.append(model.intercept_)
        
        beta_matrix = np.array(beta_matrix)
        intercept = np.array(intercept).reshape(len(intercept), 1)

        # Select observation and scale by intercept
        Y_obs = np.array(self.observation_data.T).reshape(-1, 1)
        Y_scaled = Y_obs - intercept
    

        # Compute posterior covariance
        Sigma_post_inv = (beta_matrix.T @ np.linalg.inv(full_error) @ beta_matrix) + np.linalg.inv(self.Sigma_0)
        self.Sigma_post = np.linalg.inv(Sigma_post_inv)
        
        # Compute posterior mean
        self.Mu_post = self.Sigma_post @ (beta_matrix.T @ np.linalg.inv(full_error) @ Y_scaled + np.linalg.inv(self.Sigma_0) @ self.mu_0)
        
       