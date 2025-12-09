import numpy as np
import pandas as pd

def build_local_emulator(mu_pred,
                         X_prior,         # full prior parameter matrix (n_samples × n_params)
                         Y_prior,         # full prior outputs (n_samples × n_outputs)
                         k_neighbors=200):

    # Remove intercept
    theta_pred = mu_pred[1:]  

    # Convert to numpy arrays if needed
    if isinstance(X_prior, pd.DataFrame):
        X_prior_array = X_prior.iloc[:, :len(theta_pred)].values
    else:
        X_prior_array = X_prior[:, :len(theta_pred)]
    
    if isinstance(Y_prior, pd.DataFrame):
        Y_prior_array = Y_prior.values
    else:
        Y_prior_array = Y_prior

    # --- 1. find KNN in parameter space ---
    distances = np.linalg.norm(X_prior_array - theta_pred, axis=1)
    idx = np.argsort(distances)[:k_neighbors]

    X_local = X_prior_array[idx]
    Y_local = Y_prior_array[idx]

    # --- 2. local regressions per output key ---
    n_outputs = Y_local.shape[1]
    n_params = X_local.shape[1]

    B = np.zeros((n_outputs, n_params))
    B0 = np.zeros(n_outputs)
    mse_list = np.zeros(n_outputs)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    for j in range(n_outputs):

        y = Y_local[:, j]
        model = LinearRegression().fit(X_local, y)
        y_pred = model.predict(X_local)

        B[j, :] = model.coef_
        B0[j] = model.intercept_
        mse_list[j] = mean_squared_error(y, y_pred)

    Sigma_emu = np.diag(mse_list)

    return B, B0, Sigma_emu