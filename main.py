import json
from pipeline.simulate_data import simulate_data
from pipeline.analyse_giessen import analyse_giessen
from pipeline.build_emulator import build_emulator
from pipeline.simulate_posterior import simulate_posterior
from pipeline.calibrate import calibrate
import argparse
def run_pipeline(config):
    steps = config.get("steps", ["1", "2", "3", "4", "5", "6"])
    nsamples = config.get("nsamples", 5000)

    if "1" in steps:
        print("Step 1: Simulating Data")
        simulate_data(
            param_path=config.get("input_parameters"),
            n_sample=nsamples,
            output_path=config.get("output_path"),
        )

    if "2" in steps:
        print("Step 2: Analysing Giessen (resample)")
        analyse_giessen("data/input_5000_6params.csv")

    if "3" in steps:
        print("Step 3: Building Emulator")
        build_emulator()

    if "4" in steps:
        print("Step 4: Simulating Posterior Data")
        simulate_posterior()

    if "5" in steps:
        print("Step 5: Calibration")
        calibrate()

    if "6" in steps:
        print("Step 6: Final Resampling")
        analyse_giessen("outputs/posterior_simulations.csv")

    print("Pipeline complete.")

if __name__ == "__main__":
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