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

    steps = config.get("steps", ["sim",
                                 "ag",
                                 "pca",
                                 "emu",
                                 "cal",
                                 "post_sim",
                                 "post_res"])

    data_type = config.get("data_type", "synthetic")

    if data_type == "synthetic":

        print("Processing the pipeline for synthetic data.")

        n_samples = config.get("n_samples", 2048)

        # Parent folder for all simulations
        output_path = config.get("output_path")

        if not "sim" in steps:
            # Get the n_params from the config if not provided by step 1
            n_params = config.get("n_params")

            if n_params is None:
                raise ValueError("n_params must be provided in the configuration if step 1 is not being executed.")

            # Define the output directory for the current simulations
            output_dir_sims = os.path.join(output_path, f'output_{n_samples}_{n_params}_params')
            print("Simulation output directory is: ", output_dir_sims)

        os.makedirs(output_path, exist_ok=True)

        if "sim" in steps:
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

        if "ag" in steps:
            print("Step 2: Analysing Giessen (resample)")
            analyse_giessen(file_path=output_dir_sims,
                            data_type=data_type,
                            gaussian_sigmas=config.get('gaussian_sigmas'),
            )

        if "pca" in steps:
            print("Step 3: Compute PCA")

            n_pca_components = config.get("n_pca_components", 10)
            if n_pca_components is None:
                raise ValueError("n_pca_components must be provided in the configuration to run PCA.")

            compute_pca(n_samples=n_samples, 
                        n_params=n_params, 
                        n_pca_components=n_pca_components,
                        output_path=output_path,
                        data_type=data_type)

        if "emu" in steps:
            print("Step 4: Building Emulator")
            output_keys = config.get("output_keys", None)
            build_emulator(n_samples=n_samples,
                        n_params=n_params, 
                        output_path=output_path, 
                        output_file_name="waveform_resampled_all_pressure_traces_rv_with_pca.csv",
                        output_keys_red=output_keys)

        if "gsa" in steps:
            print("Step GSA: Global Sensitivity Analysis")
            sensitivity_analysis(n_samples=n_samples,
                                 n_params=n_params, 
                                 output_path=output_path,
                                 n_processes=None)

        if "cal" in steps:
            print("Step 5: Calibrating parameters using config output keys")

            output_keys = config.get("output_keys")
            if output_keys is None:
                raise ValueError("output keys must be provided in the configuration to run calibration.")
            
            include_timeseries = bool(config.get("include_timeseries"))
            
            dummy_data_dir = config.get("dummy_data_dir")
            if dummy_data_dir is None and not data_type == "real":
                raise ValueError("Dummy data directory 'dummy_data_dir', must be provided in the configuration to run calibration for synthetic data.")
            
            output_dir_bayesian, e_obs = calibrate_parameters(
                                        data_type=data_type,
                                        n_samples=n_samples,
                                        n_params=n_params,
                                        output_path=output_path,
                                        output_keys=output_keys,
                                        include_timeseries=include_timeseries,
                                        epsilon_obs_scale=config.get("epsilon_obs_scale", 0.05),
                                        dummy_data_dir=dummy_data_dir,
                                        config=config)

        if "post_sim" in steps:
            print("Step 6: Simulating posterior pressure waves.")
        
        
            if not "cal" in steps:
                output_dir_bayesian = config.get("output_dir_bayesian")
                print(f"Reading parameter file from {output_dir_bayesian} as pre-defined in the configuration file.")

            output_dir_bayesian, n_params = simulate_data(
                param_path=os.path.join('./input_parameters_jsons', config.get("input_parameters")),
                n_samples=n_samples,
                output_path=output_dir_bayesian,
                sample_parameters = False
            )

        if "post_res" in steps:
            print("Step 7: Resampling posterior pressure waves.")

            if not "post_sim" in steps:
                output_dir_bayesian = config.get("output_dir_bayesian")
                print(f"Loading posterior samples from {output_dir_bayesian} as pre-defined in the configuration file.")

            analyse_giessen(file_path=output_dir_bayesian, 
                            data_type=data_type,
                            gaussian_sigmas=config.get('gaussian_sigmas')
                            )
            dummy_data_dir = config.get("dummy_data_dir")
            print(f"Reading true waveform from {dummy_data_dir} as pre-defined in the configuration file.")
            plot_utils.plot_posterior_simulations(dummy_data_dir, output_dir_bayesian)

        print("Pipeline complete.")

    elif data_type == "real":
        print("Processing the pipeline for real data.")
        
        # Parent folder for all simulations
        output_path = config.get("output_path")
        
        # Define the output directory for the current data
        print("Data directory is: ", output_path)
        
        if "ag" in steps:
            print("Step 2: Analysing Giessen (resample)")
            analyse_giessen(file_path=output_path,
                            data_type=data_type,
                            gaussian_sigmas=config.get('gaussian_sigmas'))

        if "pca" in steps:
            print("Step 3: Compute PCA")

            n_pca_components = config.get("n_pca_components", 10)
            if n_pca_components is None:
                raise ValueError("n_pca_components must be provided in the configuration to run PCA.")

            compute_pca(n_pca_components=n_pca_components,
                        output_path=output_path,
                        data_type=data_type,)

        if "cal" in steps:
            print("Step 5: Calibrating parameters using config output keys")

            output_keys = config.get("output_keys")
            if output_keys is None:
                raise ValueError("output keys must be provided in the configuration to run calibration.")
            
            emulator_path = config.get("emulator_path")
            n_samples = config.get("n_samples")
            n_params = config.get("n_params")
            include_timeseries = bool(config.get("include_timeseries"))

            dummy_data_dir = config.get("dummy_data_dir")
            if dummy_data_dir is None and not data_type == "real":
                raise ValueError("Dummy data directory 'dummy_data_dir', must be provided in the configuration to run calibration for synthetic data.")

            calibrate_parameters(data_type=data_type,
                                 n_samples=n_samples,
                                 n_params=n_params,
                                 output_path=output_path,
                                 emulator_path=emulator_path,
                                 output_keys=output_keys,
                                 include_timeseries=include_timeseries,
                                 dummy_data_dir=dummy_data_dir,
                                 config=config)

        print("Pipeline complete.")

    else:
        raise ValueError(f"Unknown data type: {data_type}. Supported types are 'synthetic' and 'real'.")

def main():
    parser = argparse.ArgumentParser(description="Run the pipeline with a configuration file.")
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

if __name__ == "__main__":
    main()