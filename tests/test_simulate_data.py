import os
import pytest
import pandas as pd
from simulate_data import simulate_data
import tempfile

def test_simulate_data_integration():
    # Define test parameters

    with tempfile.TemporaryDirectory() as tmp_path:
        print(f"Temporary directory created at: {tmp_path}")

        param_path = "parameters_pulmonary_sensitive_summarystats.json"  # Ensure this file exists with valid parameters
        n_sample = 10
        repeat_simulations = True

    # Call the function
        simulate_data(
            param_path=param_path,
            n_sample=n_sample,
            output_path=str(tmp_path),
            repeat_simulations=repeat_simulations
        )

        # Verify that the output files are created
        input_file = os.path.join(tmp_path,f'input_{n_sample}_9params.csv')
        output_dir = os.path.join(tmp_path,f'output_{n_sample}_9params')
        bool_indices_file = os.path.join(output_dir,f'bool_indices_{n_sample}.csv')
        output_dir_pressure_traces_pat = os.path.join(output_dir,'pressure_traces_pat','all_pressure_traces.csv')
        output_dir_pressure_traces_rv = os.path.join(output_dir,'pressure_traces_rv','all_pressure_traces.csv')


        assert os.path.exists(input_file), "Input file was not created."
        assert os.path.exists(output_dir), "Output directory was not created."
        assert os.path.exists(bool_indices_file), "Bool indices file was not created."
        assert os.path.exists(output_dir_pressure_traces_pat), "PAT pressure traces file was not created."
        assert os.path.exists(output_dir_pressure_traces_rv), "RV pressure traces file was not created."


        # Optionally, check the contents of the input file
        input_data = pd.read_csv(input_file)
        assert len(input_data) == n_sample, "Input file does not contain the expected number of samples."
        print (input_data)

        # delete files to check loading simulations from disk
        os.remove(input_file)
        os.remove(bool_indices_file)
        os.remove(output_dir_pressure_traces_pat)
        os.remove(output_dir_pressure_traces_rv)

        simulate_data(
            param_path=param_path,
            n_sample=n_sample,
            output_path=str(tmp_path),
            repeat_simulations=False
        )
        # Check if the output directory is empty

        assert os.path.exists(input_file), "Input file was not created."
        assert os.path.exists(bool_indices_file), "Bool indices file was not created."
        assert os.path.exists(output_dir_pressure_traces_pat), "PAT pressure traces file was not created."
        assert os.path.exists(output_dir_pressure_traces_rv), "RV pressure traces file was not created."

