# Import necessary libraries
import numpy as np
import pandas as pd

class KalmanFilterWithEmulator:
    def __init__(self, B, B_0, Q, R, Sigma_emu, mu_0, Sigma_0):
        """
        Parameters:
        - B: (n_obs x n_params) emulator linear coefficient matrix
        - B_0: (n_obs,) emulator intercept vector
        - Q: (n_params x n_params) state transition covariance (process noise)
        - R: (n_obs x n_obs) observation noise covariance
        - Sigma_emu: (n_obs x n_obs) emulator uncertainty covariance (diagonal from emulator RSEs)
        - mu_0: (n_params,) prior mean of theta
        - Sigma_0: (n_params x n_params) prior covariance of theta
        """
        self.B = B
        self.B_0 = B_0
        self.Q = Q
        self.R = R
        self.Sigma_emu = Sigma_emu
        self.Sigma_obs_total = Sigma_emu + R

    
        # Augment mu_0 to include intercept
        self.mu = np.insert(mu_0, 0, 1)

        # Augment Sigma_0 to include intercept
        # First, add a row of ones on top
        self.Sigma_0row = np.vstack((np.zeros((1, Sigma_0.shape[1])), Sigma_0))
        # Then, add a column of ones to the left
        self.Sigma = np.hstack((np.zeros((self.Sigma_0row.shape[0], 1)), self.Sigma_0row))

        self.H = np.hstack((B_0.reshape(-1, 1), B))

        # Adjust Q to account for intercept
        self.Q_0row = np.vstack((np.zeros((1, Q.shape[1])), Q))
        # Then, add a column of ones to the left
        self.Qnew = np.hstack((np.zeros((self.Q_0row.shape[0], 1)), self.Q_0row))

        # # Save mu, Sigma, H and Qnew, B_0 and B
        # pd.DataFrame(self.mu).to_csv("kf_mu_init.csv", index=False)
        # pd.DataFrame(self.Sigma).to_csv("kf_Sigma_init.csv", index=False)
        # pd.DataFrame(self.H).to_csv("kf_H.csv", index=False)
        # pd.DataFrame(self.Qnew).to_csv("kf_Qnew.csv", index=False)
        # pd.DataFrame(self.B_0).to_csv("kf_B_0.csv", index=False)
        # pd.DataFrame(self.B).to_csv("kf_B.csv", index=False)
        
        
    def step(self, y_t):
        """
        Perform one Kalman update step given observation y_t.
        Returns posterior mean and covariance of theta at this time.
        """

        # Prediction : State transition matrix F is implicitly the identity i.e. a random walk
        mu_pred = self.mu
        Sigma_pred = self.Sigma + self.Qnew

        # Kalman gain
        S = self.H @ Sigma_pred @ self.H.T + self.Sigma_obs_total
        K = np.linalg.solve(S, (Sigma_pred @ self.H.T).T).T 

        # Innovation (residual)
        innovation = y_t - (self.H @ mu_pred)

        # Update
        self.mu = mu_pred + K @ innovation
        self.Sigma = (np.eye(len(self.mu)) - K @ self.H) @ Sigma_pred
        
        
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