import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd
import math


class BayesianCalibration:
    def __init__(self, input_prior, emulator_output, filtered_output, which_obs, 
                 epsilon_obs):
        self.input_prior = input_prior
        self.emulator_output = emulator_output
        self.epsilon_obs = epsilon_obs 
        self.filtered_output = filtered_output
        self.which_obs = which_obs        


        # Priors
        self.mu_0 = np.array(input_prior.mean())
        self.ind = input_prior.columns.get_loc("T")
        self.mu_0[self.ind] = input_prior.iloc[which_obs]['T']  
        self.mu_0 = self.mu_0.reshape(-1, 1)
        self.Sigma_0 = np.diag(input_prior.var(ddof=0))
        self.Sigma_0[self.ind, self.ind] = 0.0000001                
        
        # Parameter names
        self.param_names = input_prior.columns.to_list()

        # Model error
        self.epsilon_model = np.diag(emulator_output['RSE']**2) 
       
      
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
        Y_obs = np.array(self.filtered_output.T[self.which_obs]).reshape(-1, 1)
        Y_scaled = Y_obs - intercept


        # Compute posterior covariance
        Sigma_post_inv = (beta_matrix.T @ np.linalg.inv(full_error) @ beta_matrix) + np.linalg.inv(self.Sigma_0)
        self.Sigma_post = np.linalg.inv(Sigma_post_inv)        
        
        # Compute posterior mean
        self.Mu_post = self.Sigma_post @ (beta_matrix.T @ np.linalg.inv(full_error) @ Y_scaled + np.linalg.inv(self.Sigma_0) @ self.mu_0)
                
        
    def sample_posterior(self, n_samples):
        rg = np.random.default_rng(1)
        self.samples = rg.multivariate_normal(self.Mu_post.flatten(), self.Sigma_post, size=n_samples)  # Generate 10 samples
        self.samples_df = pd.DataFrame(self.samples)
        self.samples_df.columns = self.param_names