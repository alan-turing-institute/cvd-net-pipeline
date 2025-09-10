# Import necessary libraries
import numpy as np

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

        self.mu = mu_0
        self.Sigma = Sigma_0

    def step(self, y_t):
        """
        Perform one Kalman update step given observation y_t.
        Returns posterior mean and covariance of theta at this time.
        """
        # Prediction : State transition matrix F is implicitly the identity i.e. a random walk
        mu_pred = self.mu
        Sigma_pred = self.Sigma + self.Q

        # Kalman gain
        S = self.B @ Sigma_pred @ self.B.T + self.Sigma_obs_total
        K = Sigma_pred @ self.B.T @ np.linalg.inv(S)

        # Innovation (residual)
        innovation = y_t - (self.B @ mu_pred + self.B_0)

        # Update
        self.mu = mu_pred + K @ innovation
        self.Sigma = (np.eye(len(self.mu)) - K @ self.B) @ Sigma_pred

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
            estimates.append((mu_t.copy(), Sigma_t.copy()))
        return estimates