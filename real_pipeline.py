import json
from analyse_giessen_real import analyse_giessen_real
from compute_pca_real import compute_pca_real
from calibrate_parameters_real import calibrate_parameters_real
from utils import plot_utils
import os
import argparse

def run_pipeline(config):

    steps = config.get("steps", ["2", "3", "5"])
    
    # Parent folder for all simulations
    output_path = config.get("output_path")
    
    # Define the output directory for the current data
    print("Data directory is: ", output_path)
    
    if "2" in steps:
        print("Step 2: Analysing Giessen (resample)")
        analyse_giessen_real(output_path)

    if "3" in steps:
        print("Step 3: Compute PCA")

        n_pca_components = config.get("n_pca_components", 10)
        if n_pca_components is None:
            raise ValueError("n_pca_components must be provided in the configuration to run PCA.")

        compute_pca_real(n_pca_components=n_pca_components,
                    output_path=output_path)


    if "5" in steps:
        print("Step 5: Calibrating parameters using config output keys")

        output_keys = config.get("output_keys")
        if output_keys is None:
            raise ValueError("output keys must be provided in the configuration to run calibration.")
        
        emulator_path = config.get("emulator_path")
        nsamples = config.get("n_samples")
        n_params = config.get("n_params")
        include_timeseries = config.get("include_timeseries")
        if include_timeseries == 1:
            include_timeseries = True
            #print("Including time-series in calibraiton as specified in config file.")
        else:
            include_timeseries = False

        calibrate_parameters_real(n_samples=nsamples,
                                    n_params=n_params,
                                    output_path=output_path,
                                    emulator_path=emulator_path,
                                    output_keys=output_keys,
                                    include_timeseries=include_timeseries,
                                    config=config)

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