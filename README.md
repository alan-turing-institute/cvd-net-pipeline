```markdown
# CVDNet pipeline 

This repository contains the pipeline for Calibration from Fay Frost as part of the CVDNet project.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
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

The pipeline can be run by executing the `main.py` script. You can specify which steps to run using the `--steps` argument.

### Steps in the Pipeline
1. **Simulate Data**: Generates input and output data based on parameters.
2. **Analyze Giessen**: Performs analysis on the input data.
3. **Build Emulator**: Builds an emulator for the data.
4. **Simulate Posterior Data**: Simulates data from posterior samples.
5. **Calibration**: Calibrates the model.
6. **Final Resampling**: Performs final resampling on posterior simulations.

### Running the Pipeline

To run specific steps, use the `--steps` argument followed by the step numbers. For example:
```bash
python main.py --steps 1 2 3
```

To specify the number of samples for the simulation, use the `-nsamples` argument (default is 5000):
```bash
python main.py --steps 1 -nsamples 10000
```

If no steps are specified, the script will prompt you to provide them.

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

