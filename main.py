import argparse
from pipeline.simulate_data import simulate_data
from pipeline.analyse_giessen import analyse_giessen
from pipeline.build_emulator import build_emulator
from pipeline.simulate_posterior import simulate_posterior
from pipeline.calibrate import calibrate

def main(steps):
    if "1" in steps:
        print("Step 1: Simulating Data")
        simulate_data(
            "config/parameters_sensitive.json",
            nsamples=args.nsamples,
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
    parser = argparse.ArgumentParser(description="Run selected steps in the pipeline.")
    parser.add_argument(
        "--steps", nargs="+", choices=["1", "2", "3", "4", "5", "6"],
        help="Steps to run, e.g. --steps 1 2 3 or --steps 4 5", default=["1", "2", "3", "4", "5", "6"]
    )
    parser.add_argument('-nsamples', type=int, default=5000,help='Number of samples')
    args = parser.parse_args()


    main(args.steps)
