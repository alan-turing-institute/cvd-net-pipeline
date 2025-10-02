import os
import pandas as pd
import pytest
from cvdnet_pipeline.simulate_data import simulate_data
import tempfile
import shutil

# Tolerance for floating point comparisons
RTOL_TOLERANCE = 1e-2

from test_constants import (
    DEFAULT_N_SAMPLES,
    DEFAULT_N_PARAMS,
    DEFAULT_EPSILON_OBS_SCALE
)

# Common test parameters
PARAM_PATH = os.path.join('./tests/input_parameters_jsons_for_tests', 
                         "parameters_pulmonary_sensitive_summarystats.json")
N_SAMPLES = DEFAULT_N_SAMPLES


# @pytest.mark.skip(reason="Temporarily disabled")
def test_simulate_data_basic():
    """Test the basic simulate_data() function with repeat_simulations=True."""
    
    with tempfile.TemporaryDirectory() as tmp_path:
        print(f"Temporary directory created at: {tmp_path}")

        repeat_simulations = True

        # Call the function
        simulate_data(
            param_path=PARAM_PATH,
            n_samples=N_SAMPLES,
            output_path=str(tmp_path),
            repeat_simulations=repeat_simulations
        )

        # Verify that the output files are created
        input_file                     = os.path.join(tmp_path,f'input_{N_SAMPLES}_{DEFAULT_N_PARAMS}_params.csv')
        output_dir_sims                = os.path.join(tmp_path,f'output_{N_SAMPLES}_{DEFAULT_N_PARAMS}_params')
        bool_indices_file              = os.path.join(output_dir_sims,f'bool_indices_{N_SAMPLES}.csv')
        output_dir_pressure_traces_pat = os.path.join(output_dir_sims,'pressure_traces_pat','all_pressure_traces.csv')
        output_dir_pressure_traces_rv  = os.path.join(output_dir_sims,'pressure_traces_rv','all_pressure_traces.csv')
        assert os.path.exists(input_file), "Input file was not created."
        assert os.path.exists(output_dir_sims), "Simulations Output directory was not created."
        assert os.path.exists(bool_indices_file), "Bool indices file was not created."

        # Optionally, check the contents of the input file
        input_data = pd.read_csv(input_file)
        assert len(input_data) == N_SAMPLES, "Input file does not contain the expected number of samples."

        # Compare the input file to the input file in the expected_outputs directory
        expected_input_file_path = os.path.join('./tests/known_good_outputs/synthetic_data/',
                                           f'input_{N_SAMPLES}_{DEFAULT_N_PARAMS}_params.csv')
        expected_input_data = pd.read_csv(expected_input_file_path)
        pd.testing.assert_frame_equal(input_data, expected_input_data)

        # Compare the output files to the expected output files
        expected_output_dir = os.path.join('./tests/known_good_outputs/synthetic_data/',
                                            f'output_{N_SAMPLES}_{DEFAULT_N_PARAMS}_params/')
        expected_pressure_traces_pat = pd.read_csv(os.path.join(expected_output_dir,
                                                    'pressure_traces_pat',
                                                    'all_pressure_traces.csv'))
        expected_pressure_traces_rv = pd.read_csv(os.path.join(expected_output_dir,
                                                    'pressure_traces_rv',
                                                    'all_pressure_traces.csv'))

        resulting_pressure_traces_pat = pd.read_csv(output_dir_pressure_traces_pat)
        resulting_pressure_traces_rv = pd.read_csv(output_dir_pressure_traces_rv)

        pd.testing.assert_frame_equal(resulting_pressure_traces_pat, 
                                      expected_pressure_traces_pat,
                                      check_exact=False,
                                      rtol=RTOL_TOLERANCE)
        pd.testing.assert_frame_equal(resulting_pressure_traces_rv, 
                                      expected_pressure_traces_rv,
                                      check_exact=False,
                                      rtol=RTOL_TOLERANCE)


def test_simulate_data_calibrated():
    """Test the simulate_data() function on calibrated parameters with sample_parameters=False."""
    
    with tempfile.TemporaryDirectory() as tmp_path:
        print(f"Temporary directory created at: {tmp_path}")

        # Create the bayesian calibration results structure in temporary directory
        output_dir_bayesian = os.path.join(tmp_path, 
                                           f'output_{N_SAMPLES}_{DEFAULT_N_PARAMS}_params',
                                           'bayesian_calibration_results',
                                           '17_output_keys/calibration_20250917_135136')
        os.makedirs(output_dir_bayesian, exist_ok=True)

        # Copy the expected output folders and files for comparison
        source_calibration_dir = os.path.join('./tests/known_good_outputs/',
                                             f'synthetic_data/output_{N_SAMPLES}_{DEFAULT_N_PARAMS}_params/',
                                             'bayesian_calibration_results',
                                             '17_output_keys/calibration_20250917_135136')
        
        # Copy only the specific folders and files needed for the test
        for folder_name in ['pressure_traces_pat', 'pressure_traces_rv']:
            source_folder = os.path.join(source_calibration_dir, folder_name)
            if os.path.exists(source_folder):
                dest_folder = os.path.join(output_dir_bayesian, folder_name)
                shutil.copytree(source_folder, dest_folder)
        
        # Copy the cleaned_posterior_samples.csv file needed by simulate_data
        source_csv = os.path.join(source_calibration_dir, 'cleaned_posterior_samples.csv')
        if os.path.exists(source_csv):
            dest_csv = os.path.join(output_dir_bayesian, 'cleaned_posterior_samples.csv')
            shutil.copy2(source_csv, dest_csv)

        # Run the test for calibrated parameters
        output_dir_sims, n_params = simulate_data(
            param_path=PARAM_PATH,
            n_samples=N_SAMPLES,
            output_path=output_dir_bayesian,
            sample_parameters=False
        )

        # Check that the 'posterior_simulations' folder is created
        assert os.path.exists(os.path.join(output_dir_bayesian, 
                                           'posterior_simulations')), "Posterior simulations directory was not created."

        # Compare the output files to the expected output files
        expected_output_dir = os.path.join('./tests/known_good_outputs/synthetic_data/',
                                            f'output_{N_SAMPLES}_{DEFAULT_N_PARAMS}_params/',
                                            'bayesian_calibration_results/17_output_keys/calibration_20250917_135136/')
        expected_pressure_traces_pat = pd.read_csv(os.path.join(expected_output_dir,
                                                    'pressure_traces_pat',
                                                    'all_pressure_traces.csv'))
        expected_pressure_traces_rv = pd.read_csv(os.path.join(expected_output_dir,
                                                    'pressure_traces_rv',
                                                    'all_pressure_traces.csv'))

        resulting_pressure_traces_pat = pd.read_csv(os.path.join(output_dir_bayesian,'pressure_traces_pat','all_pressure_traces.csv'))
        resulting_pressure_traces_rv = pd.read_csv(os.path.join(output_dir_bayesian,'pressure_traces_rv','all_pressure_traces.csv'))

        pd.testing.assert_frame_equal(resulting_pressure_traces_pat, 
                                      expected_pressure_traces_pat,
                                      check_exact=False,
                                      rtol=RTOL_TOLERANCE)
        pd.testing.assert_frame_equal(resulting_pressure_traces_rv, 
                                      expected_pressure_traces_rv,
                                      check_exact=False,
                                      rtol=RTOL_TOLERANCE)
