# CVDNet pipeline

This repository contains the pipeline for the rapid calibration of the Korakianitis and Shi cardiovascular model via linear emulators. 

Authors: Fay Frost, Levan Bokeria, Camila Rangel-Smith, Max Balmus

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alan-turing-institute/cvd-net-pipeline
   cd cvd-net-pipeline
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install

You can install the dependencies using the `pyproject.toml` file:
   ```bash
   pip install .
   ```

## Usage

The pipeline can be run by executing the `cvdnet-pipe` command. You must specify a configuration file in JSON format using the `--config` argument. For example, to run the pipeline using the the config file named `synthetic.json` you should use the command
`cvdnet-pipe --config config/synthetic.json`.

### Configuration File

The configuration file should define which pipeline steps to run and other arguements such as the number of samples and which outputs should used for calibration. Below is an example configuration file:

```json
{
    "data_type": "synthetic",
    "input_parameters": "parameters_sensitive_pca123_9.json",
    "output_path": "output_synthetic",
    "steps": [
        "sim",
        "ag",
        "pca",
        "emu",
        "cal"
    ],
    "n_pca_components": 10,
    "n_params": 9,
    "n_samples": 100,
    "gaussian_sigmas": [
        1e-05,
        1e-05,
        1e-05
    ],
    "output_keys": [
        "edp",
        "dia",
        "epad",
        "eivc",
        "sys",
        "esp",
        "a_epad",
        "P_max",
        "EF",
        "Ees/Ea",
        "iT"
    ],
    "include_timeseries": 0,
    "epsilon_obs_scale": 0.05,
    "output_dir_bayesian": "output_synthetic/output_4096_11_params/bayesian_calibration_results/11_output_keys/calibration_20250818_141427",
    "dummy_data_dir": "dummy_data"
}

```
```data_type```: Choose 'synthetic' or 'real' for synthetic analysis or calibrating real data.

```input_parameters```: The json file containting parameters and their ranges to sample from. 

```output_path```: The path you want output to be saved to.

```steps```: The steps you want to run. Choose from below.

```n_pca_components```: The number of pca components to compute.

```n_params```: The number of non-fixed parameters sampled.

```n_samples```: The number of samples you want to run.

```gaussian_sigmas```: Parameters which control peak identification in waveform analysis - best not to change.

```output_keys```: The outputs you want to calibrate on.

```include_timeseries```: Whether the timeseries waveform should be included in the calibration 0/1 for false/true.

```epsilon_obs_scale```: Observation error

```output_dir_bayesian```: The directory where calibration files are saved if you are only running steps 6 onwards.  

```dummy_data```: The dummy data directory used in synthetic calibration. 

### Running the Pipeline

To run the pipeline, specify the configuration file as follows:

```bash
cvdnet-pipe --config config/synthetic.json
```

### Steps in the Pipeline
1. **Simulate Data** ("sim"): Generates input and output data based on parameters given in JSON file.
2. **Analyze Giessen** ("ag"): Performs analysis on the simulated pressure waveform data.
3. **Compute PCA** ("pca"): Performs a PCA on the output data.
4. **Build Emulator** ("emu"): Builds an emulator for the data.
5. **Sensitivity Analysis** ("gsa"): Performs a sensitivty analysis of the model. 
6.  **Calibration** ("cal"): Calibrates the model using repeated application of the inverse problem - for synthetic analysis the model is calibrated on one observation from dummy data.
7.  **Kalman Filter** ("kf"): Calibrates the model using a Kalman Filter.
8.  **Simulate Posterior Data** ("post_sim"): Simulates data from posterior samples - for synthetic analysis only.
9. **Reconstruct waveform** ("post_res"): Plots posterior versus true waveform for a chosen observation in dummy data  - for synthetic analysis only.

### Example

To run specific steps, modify the `steps` field in the configuration file. For example, to run steps "sim", "ag", and "pca", use the following within your configuration:

```json
{
    "steps": ["sim", "ag", "pca"],

}
```


## Project Structure

- `run_pipeline.py`: Entry point for running the pipeline.

- `src/cvdnet_pipeline/`: Contains the modules for each step of the pipeline.
  - `simulate_data.py`: Simulates input and output data.
  - `analyse_giessen.py`: Analyzes the data.
  - `compute_pca.py`: Computes pca of the waveform.
  - `build_emulator.py`: Builds the emulator.
  - `sensitivity_analysis`: Performs a global sensitivity analysis of the model using linear emulators.
  - `calibrate.py`: Calibrates the model using the inverse problem.
  - `kalman_filter_giessen`: Calibrates the model using the Kalman Filter.
   
- `src/cvdnet_pipeline/utils/`: Contains scripts which assist in running of the modules above.
   - `bayesian_calibration.py`: Inverse problem calibration.
   - `constants.py`: List of valid steps.
   - `helper_functions.py: Miscellaneous functions.
   - `kf_emulator.py`: Kalman filter.
   - `plot_utils.py`: Plotting functions.


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

