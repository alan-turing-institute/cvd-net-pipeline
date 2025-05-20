import matplotlib.pyplot as plt
import numpy as np
import os

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

