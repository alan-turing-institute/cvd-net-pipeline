import os
import pytest
import pandas as pd
from simulate_data import simulate_data
import tempfile

def test_simulate_data():
    # Define test parameters

    with tempfile.TemporaryDirectory() as tmp_path:
        print(f"Temporary directory created at: {tmp_path}")

        param_path = "parameters_pulmonary_sensitive_summarystats.json"  # Ensure this file exists with valid parameters
        n_samples = 64
        repeat_simulations = True

        # Call the function
        simulate_data(
            param_path=param_path,
            n_samples=n_samples,
            output_path=str(tmp_path),
            repeat_simulations=repeat_simulations
        )

        # Verify that the output files are created
        input_file = os.path.join(tmp_path,f'input_{n_samples}_9params.csv')
        output_dir_sims = os.path.join(tmp_path,f'output_{n_samples}_9params')
        bool_indices_file = os.path.join(output_dir_sims,f'bool_indices_{n_samples}.csv')
        output_dir_pressure_traces_pat = os.path.join(output_dir_sims,'pressure_traces_pat','all_pressure_traces.csv')
        output_dir_pressure_traces_rv = os.path.join(output_dir_sims,'pressure_traces_rv','all_pressure_traces.csv')


        assert os.path.exists(input_file), "Input file was not created."
        assert os.path.exists(output_dir_sims), "Simulations Output directory was not created."
        assert os.path.exists(bool_indices_file), "Bool indices file was not created."


        # Optionally, check the contents of the input file
        input_data = pd.read_csv(input_file)
        assert len(input_data) == n_samples, "Input file does not contain the expected number of samples."

        # Compare the input file to the input file in the expected_outputs directory
        expected_input_file_path = os.path.join('./tests/expected_outputs/simulate_data_module',
                                           f'output_{n_samples}_9params/',
                                           f'input_{n_samples}_9params.csv')
        expected_input_data = pd.read_csv(expected_input_file_path)
        pd.testing.assert_frame_equal(input_data, expected_input_data)

        # Compare the output files to the expected output files
        expected_output_dir = os.path.join('./tests/expected_outputs/simulate_data_module',
                                            f'output_{n_samples}_9params/')
        expected_pressure_traces_pat = pd.read_csv(os.path.join(expected_output_dir,
                                                    'pressure_traces_pat',
                                                    'all_pressure_traces.csv'))
        expected_pressure_traces_rv = pd.read_csv(os.path.join(expected_output_dir,
                                                    'pressure_traces_rv',
                                                    'all_pressure_traces.csv'))

        resulting_pressure_traces_pat = pd.read_csv(output_dir_pressure_traces_pat)
        resulting_pressure_traces_rv = pd.read_csv(output_dir_pressure_traces_rv)

        pd.testing.assert_frame_equal(resulting_pressure_traces_pat, expected_pressure_traces_pat)
        pd.testing.assert_frame_equal(resulting_pressure_traces_rv, expected_pressure_traces_rv)


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
        output_dir_bayesian = './tests/expected_outputs/simulate_data_module/output_64_9params/bayesian_calibration_results/17_output_keys/calibration_20250604_100806'

        output_dir_sims, n_params = simulate_data(
            param_path=param_path,
            n_samples=n_samples,
            output_path=output_dir_bayesian,
            sample_parameters = False
        )

        # Check that the 'posterior_simulations' folder is created
        posterior_simulations_dir = os.path.join(output_dir_bayesian, 'posterior_simulations')
        assert os.path.exists(posterior_simulations_dir), "Posterior simulations directory was not created."

        # Compare the results with expected results

        # Load the expected results
        expected_calibration_results_dir = './tests/expected_outputs/calibrate_parameters_module/output_64_9params/bayesian_calibration_results/17_output_keys/calibration_20250604_100806'
        expected_posterior_covariance = pd.read_csv(os.path.join(expected_calibration_results_dir, 'posterior_covariance.csv'))
        expected_posterior_mean       = pd.read_csv(os.path.join(expected_calibration_results_dir, 'posterior_mean.csv'))
        expected_posterior_samples    = pd.read_csv(os.path.join(expected_calibration_results_dir, 'posterior_samples.csv'))

        # Load the actual results
        posterior_covariance = pd.read_csv(os.path.join(posterior_simulations_dir, 'posterior_covariance.csv'))
        posterior_mean       = pd.read_csv(os.path.join(posterior_simulations_dir, 'posterior_mean.csv'))
        posterior_samples    = pd.read_csv(os.path.join(posterior_simulations_dir, 'posterior_samples.csv'))

        # Compare the dataframes
        pd.testing.assert_frame_equal(posterior_covariance, expected_posterior_covariance)
        pd.testing.assert_frame_equal(posterior_mean, expected_posterior_mean)
        pd.testing.assert_frame_equal(posterior_samples, expected_posterior_samples)