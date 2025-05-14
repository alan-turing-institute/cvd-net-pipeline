# CVDNet pipeline

This repository contains the pipeline for Calibration from Fay Frost as part of the CVDNet project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alan-turing-institute/cvd-net-pipeline
   cd cvd-net-src
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

The configuration file should define the steps to run and other parameters such as the number of samples. Below is an example configuration file:

```json
{
    "steps": ["1", "2", "3"],
    "nsamples": 10000,
    "input_parameters": "config/parameters_sensitive.json",
    "output_path": "data/output"
}
```

### Running the Pipeline

To run the pipeline, specify the configuration file as follows:

```bash
python main.py --config config/pipeline_config.json
```

### Steps in the Pipeline
1. **Simulate Data**: Generates input and output data based on parameters.
2. **Analyze Giessen**: Performs analysis on the input data.
3. **Build Emulator**: Builds an emulator for the data.
4. **Simulate Posterior Data**: Simulates data from posterior samples.
5. **Calibration**: Calibrates the model.
6. **Final Resampling**: Performs final resampling on posterior simulations.

### Example

To run specific steps, modify the `steps` field in the configuration file. For example, to run steps 1, 2, and 3, use the following configuration:

```json
{
    "steps": ["1", "2", "3"],
    "nsamples": 5000,
    "input_parameters": "config/parameters_sensitive.json",
    "output_path": "data/output"
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