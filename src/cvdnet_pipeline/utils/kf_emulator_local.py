# Import necessary libraries
import numpy as np
import pandas as pd
from cvdnet_pipeline.build_local_emulator import build_local_emulator

class KalmanFilterWithLocalEmulator:
    def __init__(self, X_prior, Y_prior, observation_data, Q, R, mu_0, Sigma_0,
                 k_neighbors=200):
        """
        Parameters:
            - X_prior: (n_samples x n_params) prior parameter samples
            - Y_prior: (n_samples x n_obs) prior output samples
            - Q: (n_params x n_params) state transition covariance (process noise)
            - R: (n_obs x n_obs) observation noise covariance
            - mu_0: (n_params,) prior mean of theta
            - Sigma_0: (n_params x n_params) prior covariance of theta
            - k_neighbors: number of neighbours for local emulator
        """
        self.X_prior = X_prior
        self.Y_prior = Y_prior
        self.observation_data = observation_data
        self.Q = Q
        self.R = R
        self.mu = mu_0
        self.Sigma = Sigma_0
        self.k = k_neighbors



        # Augment mu_0 to include intercept
        self.mu = np.insert(mu_0, 0, 1)

        # Augment Sigma_0 to include intercept
        # First, add a row of ones on top
        self.Sigma_0row = np.vstack((np.zeros((1, Sigma_0.shape[1])), Sigma_0))
        # Then, add a column of ones to the left
        self.Sigma = np.hstack((np.zeros((self.Sigma_0row.shape[0], 1)), self.Sigma_0row))

        # Adjust Q to account for intercept
        self.Q_0row = np.vstack((np.zeros((1, Q.shape[1])), Q))
        # Then, add a column of ones to the left
        self.Qnew = np.hstack((np.zeros((self.Q_0row.shape[0], 1)), self.Q_0row))

    def step(self, y_t):
        """
        Perform one Kalman update step given observation y_t.
        Returns posterior mean and covariance of theta at this time.
        """

        # Prediction
        mu_pred = self.mu
        Sigma_pred = self.Sigma + self.Qnew

        # --- Build local emulator here ---
        B_t, B0_t, Sigma_emu_t = build_local_emulator(
            mu_pred,
            self.X_prior,
            self.Y_prior,
            self.k
        )

        H_t = np.hstack((B0_t.reshape(-1, 1), B_t))
        Sigma_obs_total = self.R + Sigma_emu_t

        # Kalman update using H_t
        S = H_t @ Sigma_pred @ H_t.T + Sigma_obs_total
        K = Sigma_pred @ H_t.T @ np.linalg.inv(S)
        innovation = y_t - (H_t @ mu_pred)

        self.mu = mu_pred + K @ innovation
        self.Sigma = (np.eye(len(self.mu)) - K @ H_t) @ Sigma_pred

        return self.mu, self.Sigma
    

    def run(self, Y):
        """
        Run the filter on a sequence of observations.
        Y: (n_timesteps x n_obs) array of observations
        Returns: list of (mu_t, Sigma_t) at each time step
        """
        estimates = []
        for y_t in Y:
            mu_t, Sigma_t = self.step(y_t)
            estimates.append((mu_t[1:].copy(), Sigma_t[1:, 1:].copy()))
        return estimates