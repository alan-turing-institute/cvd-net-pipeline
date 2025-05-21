import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  PowerTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_simulated_traces(simulated_traces, output_path):
    def save_plot(x, y, xlabel, ylabel, title, filename):
        """Helper function to create and save a plot."""
        fig, ax = plt.subplots()
        for indices in range(len(simulated_traces)):
            if not isinstance(simulated_traces[indices], bool):
                ax.plot(x(indices), y(indices), label=f'Realisation {indices}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.savefig(filename)
        plt.close(fig)

    # Create output directory for figures
    output_path_figures = os.path.join(output_path, "figures")
    os.makedirs(output_path_figures, exist_ok=True)

    # Plot 1: Pressure Transients in Arterial Tree
    save_plot(
        x=lambda i: simulated_traces[i]['T'] - simulated_traces[i]['T'].iloc[0],
        y=lambda i: simulated_traces[i]['p_pat'],
        xlabel='Time (seconds)',
        ylabel='Pressure (mmHg)',
        title='Pressure Transients in Arterial Tree',
        filename=os.path.join(output_path_figures, 'pressure_transients_arterial_tree_1.png')
    )

    # Plot 2: Resampled Pressure Transients in Arterial Tree
    save_plot(
        x=lambda i: list(range(100)),
        y=lambda i: np.interp(
            np.linspace(
                simulated_traces[i]['T'].iloc[0],
                simulated_traces[i]['T'].iloc[-1],
                100
            ),
            simulated_traces[i]['T'],
            simulated_traces[i]['p_pat']
        ),
        xlabel='Time index',
        ylabel='Pressure (mmHg)',
        title='Resampled Pressure Transients in Arterial Tree',
        filename=os.path.join(output_path_figures, 'pressure_transients_arterial_tree_100.png')
    )

    # Plot 3: Resampled Pressure Transients in RV
    save_plot(
        x=lambda i: list(range(100)),
        y=lambda i: np.interp(
            np.linspace(
                simulated_traces[i]['T'].iloc[0],
                simulated_traces[i]['T'].iloc[-1],
                100
            ),
            simulated_traces[i]['T'],
            simulated_traces[i]['p_rv']
        ),
        xlabel='Time index',
        ylabel='Pressure (mmHg)',
        title='Resampled Pressure Transients in RV',
        filename=os.path.join(output_path_figures, 'pressure_transients_RV.png')
    )


def plot_pressure_transients_arterial_tree(input_traces, output_path):
    """
    Plot pressure transients in the arterial tree and save the figures.

    Parameters:
        simulated_traces (list): List of simulated traces.
        output_path (str): Path to save the figures.
    """

    output_path_figures = os.path.join(output_path,"figures")
    os.makedirs(output_path_figures, exist_ok=True)

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
            

def plot_pca_histogram(X_pca, output_path, n_pca_components=10):

    output_path_figures = os.path.join(output_path,"figures")

    try:
        X_pca.hist(figsize=(15, 13), layout=(5, 2), alpha=0.7, color='orange', bins=30)
    except Exception:
        X_pca.hist(figsize=(15, 13), layout=(5, 2), alpha=0.7, color='orange')
    plt.suptitle(f'Histograms of the First 1{n_pca_components} Principal Components')
    plt.savefig(f'{output_path_figures}/histograms_pca.png')    