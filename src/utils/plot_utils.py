import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  PowerTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import seaborn as sns
from scipy.stats import norm

import matplotlib.pyplot as plt
import numpy as np
import math
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
        fig.tight_layout()
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

    fig.tight_layout()

    # Display the plot
    plt.savefig(f'{output_path_figures}/pressure_transients_arterial_tree_good_traces.png')


def plot_pca_explained_variance(pipeline, output_path):
    pipeline : Pipeline
    pca = pipeline.named_steps['pca']

    output_path_figures = os.path.join(output_path,"figures")
    os.makedirs(output_path_figures, exist_ok=True)

    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    axs[0].bar(grid, explained_variance_ratio, log=True)
    axs[0].set(xlabel="Component", title="% Explained Variance", ylim=(1e-3, 1.0))

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


def plot_pca_transformed(pipeline, X, output_path):
    pipeline : Pipeline

    output_path_figures = os.path.join(output_path,"figures")
    os.makedirs(output_path_figures, exist_ok=True)

    signals_pca = pipeline.transform(X)

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
    
    plt.tight_layout()
    plt.suptitle(f'Histograms of the First {n_pca_components} Principal Components')
    plt.savefig(f'{output_path_figures}/histograms_pca.png')    


def plot_posterior_distributions(input, mu_0, Sigma_0, Mu_post, Sigma_post, which_obs, param_names, output_path):

    output_path_figures = os.path.join(output_path,"figures")
    os.makedirs(output_path_figures, exist_ok=True)

    prior_means = mu_0.flatten()
    prior_stds = np.sqrt(np.diag(Sigma_0))
    posterior_means = Mu_post.flatten()
    posterior_stds = np.sqrt(np.diag(Sigma_post))
    true_values = input.iloc[which_obs].values
    
    
    fig, axes = plt.subplots(2, math.ceil(len(param_names)/2), figsize=(18, 8))  
    axes = axes.flatten()  # Flatten to 1D array
    for i, ax in enumerate(axes[:len(param_names)]):  # Only iterate over valid axes
    
        # Define x-range based on prior and posterior means
        x_min = min(prior_means[i] - 3 * prior_stds[i], posterior_means[i] - 3 * posterior_stds[i])
        x_max = max(prior_means[i] + 3 * prior_stds[i], posterior_means[i] + 3 * posterior_stds[i])
        x = np.linspace(x_min, x_max, 100)

        # Compute PDFs
        prior_pdf = norm.pdf(x, prior_means[i], prior_stds[i])
        posterior_pdf = norm.pdf(x, posterior_means[i], posterior_stds[i])

        # Plot prior and posterior distributions
        ax.plot(x, prior_pdf, label="Prior", linestyle="dashed", color="blue")
        ax.plot(x, posterior_pdf, label="Posterior", linestyle="solid", color="red")

        # Plot true value as a vertical line
        ax.axvline(true_values[i], color="green", linestyle="dotted", label="True Value")

        # Labels and title
        ax.set_title(param_names[i])
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()

    fig.tight_layout()
    plt.suptitle(f'Posterior Distributions of Calibrated Parameters')
    plt.savefig(f'{output_path_figures}/posterior_distributions_calibrated_params.png')    


def plot_posterior_covariance_matrix(Sigma_0, Sigma_post, param_names, output_path):
        
    output_path_figures = os.path.join(output_path,"figures")
    os.makedirs(output_path_figures, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(Sigma_0, annot=True, fmt=".3f", cmap="RdBu", xticklabels=param_names, yticklabels=param_names, ax=axes[0])
    axes[0].set_title("Prior Covariance Matrix")
    sns.heatmap(Sigma_post, annot=True, fmt=".4f", cmap="PiYG", xticklabels=param_names, yticklabels=param_names, ax=axes[1])
    axes[1].set_title("Posterior Covariance Matrix")

    fig.tight_layout()
    plt.suptitle(f'Posterior Covariance Matrix')
    plt.savefig(f'{output_path_figures}/posterior_covariance_matrix.png') 

def plot_posterior_simulations(output_dir_sims, output_dir_bayesian, e_obs_scale):
    
    true_waveforms = pd.read_csv(f"{output_dir_sims}/waveform_resampled_all_pressure_traces_rv.csv")
    posterior_waveforms = pd.read_csv(f"{output_dir_bayesian}/waveform_resampled_all_pressure_traces_rv.csv")
    
    # Ground truth waveform
    which_obs = 3
    y_true = pd.Series(true_waveforms.iloc[which_obs, :101].values)

    # Posterior waveforms
    samples = posterior_waveforms.iloc[:, :101].values # Shape (100,101)
    

    # Compute and plot the mean waveform
    mean_waveform = samples.mean(axis=0)
    var_waveform = samples.var(axis=0) + 1e-6  # Adding a small constant to avoid division by zero
    
    
    fig, ax = plt.subplots(figsize=(10, 5))
    output_path_figures = os.path.join(output_dir_bayesian, "figures")
    os.makedirs(output_path_figures, exist_ok=True)

    
    
    # Inputs
    y_obs = y_true.values                    # shape: (101,)
    posterior_preds = samples               # shape: (100, 101)
    S, T = posterior_preds.shape            # S = 100, T = 101

    # Set Gaussian likelihood standard deviation (fixed)
    sigma = 1 # Dynamically adjust from observation model? Or is it always 1 as it is only concerned with waveform even if calibrated on something else?
    

    # Compute log pointwise predictive density
    log_likelihoods = -0.5 * np.log(2 * np.pi * sigma**2) \
                    - ((y_obs - posterior_preds)**2) / (2 * sigma**2)
    

    
    # log_likelihoods shape: (100, 101)
    # Average over posterior samples (axis 0), then sum over timepoints
    lppd =  np.sum(np.log(np.mean(np.exp(log_likelihoods), axis=0)))
    nlpd = -lppd

    # p_WAIC: sum of variances of log-likelihoods across posterior samples
    p_waic = np.sum(np.var(log_likelihoods, axis=0, ddof=1))  # scalar

    # WAIC computation
    waic = -2 * (lppd - p_waic)

    # Compute RMSE
    sqe = (y_obs - mean_waveform) ** 2
    rmse = np.sqrt(sqe.mean(axis=0))
    
    
    
    # Plot all waveforms in faded orange
    for j in range(samples.shape[0]):
        ax.plot(samples[j, :], color='orange', alpha=0.001)
    
    # Plot y_true
    ax.plot(y_true.values, label="True Waveform", color='c', linewidth=2)

    # Plot mean waveform
    ax.plot(mean_waveform, color='darkorange', linewidth=1.5, label="Mean Calibrated Waveform")

    ax.set_xticks(np.arange(0, 110, 10))
    ax.set_xlabel("Time Index")
    ax.set_title(f"Posterior Simulations\nRMSE = {rmse:.4f}, NLPD = {nlpd:.2f}, WAIC = {waic:.2f}")
    ax.set_ylabel("Pressure (mmHg)")
    ax.legend()

    fig.suptitle("Calibrated Pressure Waveforms for Different Methods")
    fig.tight_layout()
    fig.savefig(os.path.join(output_path_figures, "posterior_simulated_waveforms.png"))

def plot_parameter_trajectories(Sigma_post,
                                posterior_means,
                                bc,
                                output_path):
    
    output_path_figures = os.path.join(output_path,"figures")
    os.makedirs(output_path_figures, exist_ok=True)

    class ResolutionController:
        def __init__(self, window_size):
            self.window_size = window_size

        def downsample(self, data):
            """Downsamples the data by averaging over non-overlapping windows."""
            if data.shape[0] < self.window_size:
                raise ValueError(f"Data has fewer than {self.window_size} time steps!")

            num_windows = data.shape[0] // self.window_size  # Compute number of full windows
            return data[:num_windows * self.window_size].reshape(num_windows, self.window_size, -1).mean(axis=1)

    # Initialize resolution controller
    window_size = 5
    res_controller = ResolutionController(window_size)

    # Define time range before downsampling
    time_range = (1, 4000)  # Specify the indices from the original data

    # Ensure posterior_variances has shape (3888, p)
    posterior_variances_corrected = np.array(Sigma_post).diagonal().reshape(1, -1)  # (1, p)
    posterior_variances_corrected = np.tile(posterior_variances_corrected, (posterior_means.shape[0], 1))  # (3888, p)


    # Slice the original data before downsampling
    posterior_means_trimmed = posterior_means[time_range[0]:time_range[1]]
    posterior_variances_trimmed = posterior_variances_corrected[time_range[0]:time_range[1]]

    # Downsample the sliced data
    posterior_means_smooth = res_controller.downsample(posterior_means_trimmed)  # (new_length, p)
    posterior_variances_smooth = res_controller.downsample(np.sqrt(posterior_variances_trimmed))  # (new_length, p)


    # Generate new time indices based on downsampling
    T_smooth = np.arange(posterior_means_smooth.shape[0]) * window_size + time_range[0]

    # Colors for different parameters
    param_names = bc.param_names
    colors = plt.cm.get_cmap('Set1', len(param_names)).colors

    # Plot each parameter on a separate subplot
    fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 8), sharex=True)

    for i in range(len(param_names)):
        mean = posterior_means_smooth[:, i]  # Smoothed mean
        std_dev = posterior_variances_smooth[:, i]  # Smoothed standard deviation

        axes[i].plot(T_smooth, mean, color=colors[i], label=param_names[i])
        axes[i].fill_between(T_smooth, mean - 2 * std_dev, mean + 2 * std_dev, color=colors[i], alpha=0.2)

        axes[i].set_ylabel('Value')
        axes[i].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        axes[i].grid()

    axes[-1].set_xlabel('Time')
    fig.suptitle(f'Parameter Trajectories (Averaged Over {window_size} Steps) [Original Range: {time_range}]')

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make space for legends on the right
    plt.show()

    fig.savefig(os.path.join(output_path_figures, "posterior_simulated_waveforms.png"))
