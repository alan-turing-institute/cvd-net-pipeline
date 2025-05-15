import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  PowerTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


def plot_pca_explained_variance(pca, output_path):

    output_path_figures = os.path.join(output_path,"figures")
    os.makedirs(output_path_figures, exist_ok=True)

    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    axs[0].bar(grid, explained_variance_ratio, log=True)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )

    # Cumulative Variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    axs[1].semilogy(grid, cumulative_explained_variance, "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", 
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    fig.tight_layout()
            
    plt.savefig(f'{output_path_figures}/pca_explained_variance.png')

def plot_pca_transformed(pca, X_scaled, output_path):

    output_path_figures = os.path.join(output_path,"figures")
    os.makedirs(output_path_figures, exist_ok=True)
        
    pipeline = Pipeline([
                    ('scl', StandardScaler()),
                    ('pca', PCA(n_components=10)),
                    ('post',   PowerTransformer())
                ])

    signals_pca = pipeline.fit_transform(X_scaled)

    fig, ax = plt.subplots(ncols=10, nrows=2, figsize=(70, 15))

    for i in range(signals_pca.shape[1]):
        temp = np.zeros(signals_pca.shape)
        temp[:, i] = signals_pca[:, i]
        
        signals_new = pipeline.inverse_transform(temp)
        
        ax[1][i].hist(signals_pca[:,i], bins=10)
        for signal in signals_new:
            ax[0][i].plot(signal)
    
    plt.savefig(f'{output_path_figures}/pca_transformed.png')
            

