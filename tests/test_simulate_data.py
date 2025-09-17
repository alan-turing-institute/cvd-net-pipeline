import os
import pandas as pd
from cvdnet_pipeline.simulate_data import simulate_data
import tempfile
import shutil

RTOL_TOLERANCE = 1e-2

from test_constants import (
    DEFAULT_N_SAMPLES,
    DEFAULT_N_PARAMS,
    DEFAULT_EPSILON_OBS_SCALE
)

def test_simulate_data():
    # Define test parameters

    with tempfile.TemporaryDirectory() as tmp_path:
        print(f"Temporary directory created at: {tmp_path}")

        param_path = os.path.join('./input_parameters_jsons_for_tests', 
                                  "parameters_pulmonary_sensitive_summarystats.json")  # Ensure this file exists with valid parameters
        n_samples = DEFAULT_N_SAMPLES
        repeat_simulations = True

        # Call the function
        simulate_data(
            param_path=param_path,
            n_samples=n_samples,
            output_path=str(tmp_path),
            repeat_simulations=repeat_simulations
        )

        # Verify that the output files are created
        input_file                     = os.path.join(tmp_path,f'input_{n_samples}_{DEFAULT_N_PARAMS}_params.csv')
        output_dir_sims                = os.path.join(tmp_path,f'output_{n_samples}_{DEFAULT_N_PARAMS}_params')
        bool_indices_file              = os.path.join(output_dir_sims,f'bool_indices_{n_samples}.csv')
        output_dir_pressure_traces_pat = os.path.join(output_dir_sims,'pressure_traces_pat','all_pressure_traces.csv')
        output_dir_pressure_traces_rv  = os.path.join(output_dir_sims,'pressure_traces_rv','all_pressure_traces.csv')
        assert os.path.exists(input_file), "Input file was not created."
        assert os.path.exists(output_dir_sims), "Simulations Output directory was not created."
        assert os.path.exists(bool_indices_file), "Bool indices file was not created."

        # Optionally, check the contents of the input file
        input_data = pd.read_csv(input_file)
        assert len(input_data) == n_samples, "Input file does not contain the expected number of samples."

        # Compare the input file to the input file in the expected_outputs directory
        expected_input_file_path = os.path.join('./tests/expected_outputs/simulate_data_module',
                                           f'output_{n_samples}_{DEFAULT_N_PARAMS}_params/',
                                           f'input_{n_samples}_{DEFAULT_N_PARAMS}_params.csv')
        expected_input_data = pd.read_csv(expected_input_file_path)
        pd.testing.assert_frame_equal(input_data, expected_input_data)

        # Compare the output files to the expected output files
        expected_output_dir = os.path.join('./tests/expected_outputs/simulate_data_module',
                                            f'output_{n_samples}_{DEFAULT_N_PARAMS}_params/')
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


        # delete files to check loading simulations from disk
        os.remove(input_file)
        os.remove(bool_indices_file)
        os.remove(output_dir_pressure_traces_pat)
        os.remove(output_dir_pressure_traces_rv)

        simulate_data(
            param_path=param_path,
            n_samples=n_samples,
            output_path=str(tmp_path),
            repeat_simulations=False
        )
        # Check if the output directory is empty

        assert os.path.exists(input_file), "Input file was not created."
        assert os.path.exists(bool_indices_file), "Bool indices file was not created."
        assert os.path.exists(output_dir_pressure_traces_pat), "PAT pressure traces file was not created."
        assert os.path.exists(output_dir_pressure_traces_rv), "RV pressure traces file was not created."

        # Run the test for calibrated parameters
        output_dir_bayesian = os.path.join('./tests/expected_outputs/calibrate_parameters_module',
                                           'synthetic_data/output_64_{DEFAULT_N_PARAMS}_params/bayesian_calibration_results',
                                           '17_output_keys/calibration_20250827_151335')

        output_dir_sims, n_params = simulate_data(
            param_path=param_path,
            n_samples=n_samples,
            output_path=output_dir_bayesian,
            sample_parameters = False
        )

        # Check that the 'posterior_simulations' folder is created
        assert os.path.exists(os.path.join(output_dir_bayesian, 'posterior_simulations')), "Posterior simulations directory was not created."

        # Compare the output files to the expected output files
        expected_output_dir = os.path.join('./tests/expected_outputs/simulate_data_module',
                                            f'output_{n_samples}_{DEFAULT_N_PARAMS}_params/',
                                            'bayesian_calibration_results/17_output_keys/calibration_20250827_151335/')
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
