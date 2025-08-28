import json
from simulate_data import simulate_data
from analyse_giessen import analyse_giessen
from compute_pca import compute_pca
from build_emulator import build_emulator
from calibrate_parameters import calibrate_parameters
from sensitivity_analysis import sensitivity_analysis
from utils import plot_utils
import os
import argparse

def run_pipeline(config):

    steps = config.get("steps", ["1", "2", "3", "4", "5"])

    n_samples = config.get("n_samples", 2000)

    # Parent folder for all simulations
    output_path = config.get("output_path")

    if not "1" in steps:
        # Get the n_params from the config if not provided by step 1
        n_params = config.get("n_params")

        if n_params is None:
            raise ValueError("n_params must be provided in the configuration if step 1 is not being executed.")

        # Define the output directory for the current simulations
        output_dir_sims = os.path.join(output_path, f'output_{n_samples}_{n_params}_params')
        print("Simulation output directory is: ", output_dir_sims)

    os.makedirs(output_path, exist_ok=True)

    if "1" in steps:
        print("Step 1: Simulating Data")

        if "n_params" in locals():
            print("Warning: n_params is pre-defined in the configuration file. It will be overwritten by the value from the simulation step.")

        if "output_dir_sims" in locals():
            print("Warning: output_dir_sims is pre-defined in the configuration file. It will be overwritten by the value from the simulation step.")

        output_dir_sims, n_params = simulate_data(
            param_path=os.path.join('./input_parameters_jsons', config.get("input_parameters")),
            n_samples=n_samples,
            output_path=output_path,
            sample_parameters=True
        )

    if "2" in steps:
        print("Step 2: Analysing Giessen (resample)")
        analyse_giessen(file_path=output_dir_sims, 
                        data_type='synthetic',
                        gaussian_sigmas=config.get('gaussian_sigmas')
        )

    if "3" in steps:
        print("Step 3: Compute PCA")

        n_pca_components = config.get("n_pca_components", 10)
        if n_pca_components is None:
            raise ValueError("n_pca_components must be provided in the configuration to run PCA.")

        compute_pca(n_samples=n_samples, 
                    n_params=n_params, 
                    n_pca_components=n_pca_components,
                    output_path=output_path,
                    data_type='synthetic')

    if "4" in steps:
        print("Step 4: Building Emulator")
        build_emulator(n_samples=n_samples,
                       n_params=n_params, 
                       output_path=output_path, 
                       output_file_name="waveform_resampled_all_pressure_traces_rv_with_pca.csv")

    if "5" in steps:
        print("Step 5: Sensitivity Analysis")
        sensitivity_analysis(n_samples=n_samples,
                             n_params=n_params, 
                             output_path=output_path)

    print("Pipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the src with a configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (JSON format)."
    )
    args = parser.parse_args()

    # Load configuration from the specified JSON file
    with open(args.config, "r") as config_file:
        config = json.load(config_file)

    run_pipeline(config)