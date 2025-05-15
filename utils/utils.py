# utils/io.py
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

def load_csv(path, drop_column=""):
    return pd.read_csv(path, usecols=lambda x: x != drop_column)

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def load_simulation(input_path):
    simulations = []
    for file in os.listdir(input_path):
        if file.startswith('all_outputs') and file.endswith('.csv'):
            df = load_csv(os.path.join(input_path, file), drop_column="time_ind")
            simulations.append(df)
    return simulations

def select_feasible_traces(simulated_traces, screen, output_path):
    # Create column headers
    headers = list(range(100)) + ['CO', 'dt', 'EF', 'dPAP', 'sPAP', 'mPAP']

    # List to collect all pressure traces
    pressure_traces_list_pat = []
    pressure_traces_list_rv = []

    for ind in range(len(simulated_traces)):
        if not isinstance(simulated_traces[ind], bool):

            # PAT pressure
            p_pat_raw = simulated_traces[ind].loc[ind]['p_pat'].values.copy()

            # RV pressure
            p_rv_raw = simulated_traces[ind].loc[ind]['p_rv'].values.copy()

            T = simulated_traces[ind].loc[ind]['T'].values.copy()
            T_resample = np.linspace(T[0], T[-1], 100)

            # Interpolate pressure for 100 timesteps from 1000
            p_pat_resampled = np.interp(T_resample, T, p_pat_raw)
            p_rv_resampled = np.interp(T_resample, T, p_rv_raw)

            # Compute CO
            q_pat = simulated_traces[ind].loc[ind]['q_pat'].values.copy()
            CO = np.sum(q_pat) * (T[1] - T[0]) / (T[-1] - T[0]) * 60. / 1000.  # L / min

            # Compute EF
            v_rv = simulated_traces[ind].loc[ind]['v_rv'].values.copy()
            EF = (np.max(v_rv) - np.min(v_rv)) / np.max(v_rv)

            # Compute dPAP, sPAP, mPAP
            dPAP = min(p_rv_raw)
            sPAP = max(p_rv_raw)
            mPAP = np.mean(p_rv_raw)

            # Record time interval, approx T (input param) / 100, there are some rounding differences due to interpolation
            tl = T_resample - simulated_traces[ind].loc[ind]['T'].iloc[0]
            dt = np.diff(tl)[0]

            # Only create array if conditions hold or screening is turned off
            if not screen or (2 < CO < 12 and 4 < dPAP < 67 and 9 < mPAP < 87 and 15 < sPAP < 140):
                # Create a 2D array for saving
                pressure_trace_pat = np.hstack((p_pat_resampled, [CO], [dt], [EF], [dPAP], [sPAP], [mPAP]))
                pressure_trace_rv = np.hstack((p_rv_resampled, [CO], [dt], [EF], [dPAP], [sPAP], [mPAP]))
                pressure_traces_list_pat.append(pressure_trace_pat)
                pressure_traces_list_rv.append(pressure_trace_rv)

    # Save the pressure traces to CSV files
                # Save individual pressure trace to CSV with headers


    # Convert the list of pressure traces to a DataFrame
    pressure_traces_df_pat = pd.DataFrame(pressure_traces_list_pat, columns=headers)
    pressure_traces_df_rv = pd.DataFrame(pressure_traces_list_rv, columns=headers)

    return pressure_traces_df_pat, pressure_traces_df_rv

def emulate_linear(input, output):
    # Input and output data
    X = input
    Y = output

    # Initialize the model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X, Y)

    # Predict the output for the test data
    y_pred = model.predict(X)

    # Compute R² score for the predictions versus actual test data
    r2 = r2_score(Y, y_pred)

    # compute MSE
    mse = mean_squared_error(Y, y_pred)

    # compute RSE
    n = len(Y)   # Number of observations
    p = X.shape[1] + 1  # Number of parameters (including intercept)
    rss = np.sum((Y - y_pred) ** 2)  # Residual Sum of Squares
    rse = np.sqrt(rss / (n - p))


    return model, r2, mse, rse