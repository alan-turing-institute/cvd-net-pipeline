import matplotlib.pyplot as plt
import numpy as np
import os

def plot_simulated_traces(simulated_traces,output_path):


    output_path_figures = os.path.join(output_path,"figures")
    os.makedirs(output_path_figures, exist_ok=True)

    fig, ax = plt.subplots()

    # Loop over all realizations
    for indices in range(len(simulated_traces)):
        if not isinstance(simulated_traces[indices], bool):
            # Adjust time and pressure trace for each realization
            t = simulated_traces[indices].loc[indices]['T'] - simulated_traces[indices].loc[indices]['T'].loc[0]  # Time adjustment
            p_pat = simulated_traces[indices].loc[indices]['p_pat']  # Pressure transient

            # Plot the pressure transient for each realization
            ax.plot(t, p_pat, label=f'Realisation {indices}')

    # Set labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Pressure (mmHg)')
    ax.set_title('Pressure Transients in Arterial Tree')


    # Display the plot
    # save the plot
    plt.savefig(f'{output_path_figures}/pressure_transients_arterial_tree_1.png')

    # Initialize the plot
    fig, ax = plt.subplots()

    # Loop over all realizations
    for indices in range(len(simulated_traces)):
        if not isinstance(simulated_traces[indices], bool):
            p_pat_raw = simulated_traces[indices].loc[indices]['p_pat'].values.copy()
            T = simulated_traces[indices].loc[indices]['T'] - simulated_traces[indices].loc[indices]['T'].loc[0]  # Time adjustment
            T = T.values.copy()
            T_resample = np.linspace(T[0], T[-1], 100)

            # Interpolate pressure for 100 timesteps from 1000
            p_pat_resampled = np.interp(T_resample, T, p_pat_raw)

            # Plot the interpolated pressure transient for each realization
            ax.plot(list(range(100)), p_pat_resampled, label=f'Realisation {indices}')

    # Set labels and title
    ax.set_xlabel('Time index')
    ax.set_ylabel('Pressure (mmHg)')
    ax.set_title('Pressure Transients in Arterial Tree')

    # Add legend to the plot
    # ax.legend()

    # Display the plot
    plt.savefig(f'{output_path_figures}/pressure_transients_arterial_tree_100.png')



    # Initialize the plot
    fig, ax = plt.subplots()

    # Loop over all realizations
    for indices in range(len(simulated_traces)):
        if not isinstance(simulated_traces[indices], bool):
            p_rv_raw = simulated_traces[indices].loc[indices]['p_rv'].values.copy()
            T = simulated_traces[indices].loc[indices]['T'] - simulated_traces[indices].loc[indices]['T'].loc[0]  # Time adjustment
            T = T.values.copy()
            T_resample = np.linspace(T[0], T[-1], 100)

            # Interpolate pressure for 100 timesteps from 1000
            p_rv_resampled = np.interp(T_resample, T, p_rv_raw)

            # Plot the interpolated pressure transient for each realization
            ax.plot(list(range(100)), p_rv_resampled, label=f'Realisation {indices}')

    # Set labels and title
    ax.set_xlabel('Time index')
    ax.set_ylabel('Pressure (mmHg)')
    ax.set_title('Pressure Transients in RV')

    # Add legend to the plot
    # ax.legend()

    # Display the plot
    plt.savefig(f'{output_path_figures}/pressure_transients_RV.png')


def plot_pressure_transients_arterial_tree(input_traces, output_path):
    """
    Plot pressure transients in the arterial tree and save the figures.

    Parameters:
        simulated_traces (list): List of simulated traces.
        output_path (str): Path to save the figures.
    """

    output_path_figures = os.path.join(output_path,"figures")
    os.makedirs(output_path_figures, exist_ok=True)

    # Create directory for figures if it doesn't exist
    os.makedirs(os.path.join(output_path, "figures"), exist_ok=True)

    fig, ax = plt.subplots()

    for ind in range(len(input_traces)):
        t = range(100)  # Time adjustment
        p_pat = input_traces.iloc[ind, :100].values  # Pressure transient

        # Plot the pressure transient for each realization
        ax.plot(t, p_pat, label=f'Realisation {ind}')

    # Set labels and title
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Pressure (mmHg)')
    ax.set_title('Pressure Transients in Arterial Tree')

    # Add legend to the plot
    # ax.legend()

    # Display the plot
    plt.savefig(f'{output_path_figures}/pressure_transients_arterial_tree_good_traces.png')

