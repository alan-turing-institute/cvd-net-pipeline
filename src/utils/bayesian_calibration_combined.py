import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd
import math


class BayesianCalibration:
    def __init__(self, input_prior, emulator_output, epsilon_obs, 
                 filtered_output=None, which_obs=None, observation_data=None):
        """
        Bayesian calibration for both synthetic and real data.
        
        For synthetic data: provide filtered_output and which_obs
        For real data: provide observation_data
        """
        self.input_prior = input_prior
        self.emulator_output = emulator_output
        self.epsilon_obs = epsilon_obs 
        
        # Determine if we're using synthetic or real data
        if observation_data is not None:
            # Real data mode
            self.observation_data = observation_data
            self.is_real_data = True
        elif filtered_output is not None and which_obs is not None:
            # Synthetic data mode
            self.filtered_output = filtered_output
            self.which_obs = which_obs
            self.is_real_data = False
        else:
            raise ValueError("Must provide either observation_data (real) or both filtered_output and which_obs (synthetic)")

        # Setup priors
        self._setup_priors()
        
        # Parameter names
        self.param_names = input_prior.loc[:, :'T'].columns.to_list()

        # Model error
        self.epsilon_model = np.diag(emulator_output['RSE']**2) 
        
        # Compute posterior
        self.compute_posterior()

    def _setup_priors(self):
        """Setup prior parameters based on data type"""
        self.mu_0 = np.array(self.input_prior.mean().loc[:'T'])
        self.mu_0 = self.mu_0.reshape(-1, 1)
        self.Sigma_0 = np.diag(self.input_prior.var().loc[:'T'])
        
        if self.is_real_data:
            # Real data: use iT from observation_data
            self.mu_0[-1, 0] = self.observation_data['iT'].iloc[0]
        else:
            # Synthetic data: use T from specific observation
            self.mu_0[-1, 0] = self.input_prior.iloc[self.which_obs]['T']
        
        self.Sigma_0[-1, -1] = 0.0000001

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

        # Select observation based on data type
        if self.is_real_data:
            Y_obs = np.array(self.observation_data.T).reshape(-1, 1)
        else:
            Y_obs = np.array(self.filtered_output.T[self.which_obs]).reshape(-1, 1)
        
        Y_scaled = Y_obs - intercept

        # Compute posterior covariance
        Sigma_post_inv = (beta_matrix.T @ np.linalg.inv(full_error) @ beta_matrix) + np.linalg.inv(self.Sigma_0)
        self.Sigma_post = np.linalg.inv(Sigma_post_inv)
        
        # Compute posterior mean
        self.Mu_post = self.Sigma_post @ (beta_matrix.T @ np.linalg.inv(full_error) @ Y_scaled + np.linalg.inv(self.Sigma_0) @ self.mu_0)

    def sample_posterior(self, n_samples):
        rg = np.random.default_rng(1)
        self.samples = rg.multivariate_normal(self.Mu_post.flatten(), self.Sigma_post, size=n_samples)
        self.samples_df = pd.DataFrame(self.samples)
        self.samples_df.columns = self.param_names