# CVDNet pipeline

This repository contains the pipeline for Calibration from Fay Frost as part of the CVDNet project.

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

The pipeline can be run by executing the `main.py` script. You must specify a configuration file in JSON format using the `--config` argument.

### Configuration File

The configuration file should define the steps to run and other parameters such as the number of samples and with outputs used for calibration. Below is an example configuration file:

```json
{
  "input_parameters": "parameters_pulmonary_sensitive_summarystats.json",
  "output_path": "output_synthetic/",
  "steps": ["1", "2", "3", "4", "5"],  
  "nsamples": 64,
  "n_pca_components": 10,
  "n_params": 9,
  "include_timeseries": 0,
  "output_keys": [
    "t_max_dpdt", "a_epad", "epad", "s_a_epad", "s_epad",
    "min_dpdt", "max_dpdt", "A_p", "P_max", "esp", "sys",
    "EF", "dia", "Ees/Ea", "iT", "PC1", "PC2", "PC3"
  ],
  "epsilon_obs_scale": 0.05,
  "output_dir_bayesian": "output_synthetic/output_5000_9params/bayesian_calibration_results/18_output_keys/calibration_20250619_135107"
}

```

```input_parameters```: The json file containting parameters and their ranges to sample from. 

```output_path```: The path you want output to be saved to.

```steps```: The steps you want to run.

```nsamples```: The number of samples you want to run.

```n_pca_components```: The number of pca components to compute.

```n_params```: The number of non-fixed parameters sampled.

```include_timeseries```: Whether the timeseries waveform should be included in the calibration 0/1 for true/false.

```output_keys```: The outputs you want to calibrate on.

```epsilon_obs_scale```: Observation error

```output_dir_bayesian```: The directory where calibration files are saved if you are only running steps 6 onwards.  

### Running the Pipeline

To run the pipeline, specify the configuration file as follows:

```bash
python main.py --config config/pipeline_config.json
```

### Steps in the Pipeline
1. **Simulate Data**: Generates input and output data based on parameters.
2. **Analyze Giessen**: Performs analysis on the simulated pressure waveform data.
3. **Compute PCA**: Performs a PCA on the output data.
4. **Build Emulator**: Builds an emulator for the data.
5. **Simulate Posterior Data**: Simulates data from posterior samples.
6. **Calibration**: Calibrates the model.
7. **Final Resampling**: Performs final resampling on posterior simulations.

### Example

To run specific steps, modify the `steps` field in the configuration file. For example, to run steps 1, 2, and 3, use the following within your configuration:

```json
{
    "steps": ["1", "2", "3"],

}
```

Then execute:

```bash
python main.py --config config/pipeline_config.json
```

## Project Structure

- `main.py`: Entry point for running the pipeline.
- `pipeline/`: Contains the modules for each step of the pipeline.
  - `simulate_data.py`: Simulates input and output data.
  - `analyse_giessen.py`: Analyzes the data.
  - `build_emulator.py`: Builds the emulator.
  - `simulate_posterior.py`: Simulates posterior data.
  - `calibrate.py`: Calibrates the model.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```
