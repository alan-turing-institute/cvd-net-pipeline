import os
import pytest
import pandas as pd
from calibrate_parameters import calibrate_parameters
import tempfile
import shutil
import glob
from tests.test_constants import (
    OUTPUT_KEYS_FOR_TESTS_FOR_SYNTHETIC_DATA_CALIBRATION, 
    DEFAULT_N_SAMPLES, 
    DEFAULT_N_PARAMS, 
    DEFAULT_EPSILON_OBS_SCALE
)

def test_calibrate_parameters():
    # Define test parameters

    with tempfile.TemporaryDirectory() as tmp_path:
        print(f"Temporary directory created at: {tmp_path}")

        output_keys = OUTPUT_KEYS_FOR_TESTS_FOR_SYNTHETIC_DATA_CALIBRATION
        n_samples = DEFAULT_N_SAMPLES
        n_params = DEFAULT_N_PARAMS

        # Copy all the expected input files from /tests/inputs_for_tests/calibrate_parameters_module/ to the temporary directory
        shutil.copytree('./tests/inputs_for_tests/calibrate_parameters_module/synthetic_data',
                        tmp_path,
                        dirs_exist_ok=True)

        calibrate_parameters(data_type="synthetic",
                             n_samples=n_samples,
                             n_params=n_params,
                             output_path=str(tmp_path),
                             output_keys=output_keys,
                             include_timeseries=False,
                             epsilon_obs_scale=DEFAULT_EPSILON_OBS_SCALE,
                             dummy_data_dir='./tests/inputs_for_tests/calibrate_parameters_module/dummy_data/',
                             config=[])

        # Compare the output files to the expected output files

        # Load the expected output ----------------------------------------------------------------------
        expected_output_dir = os.path.join(
            './tests/expected_outputs/calibrate_parameters_module',
            'synthetic_data',
            'output_64_9_params',
            'bayesian_calibration_results',
            '17_output_keys',
            'calibration_20250827_151335'
        )
        expected_posterior_covariance = pd.read_csv(os.path.join(expected_output_dir,
                                                    'posterior_covariance.csv'))
        expected_posterior_mean       = pd.read_csv(os.path.join(expected_output_dir,
                                                    'posterior_mean.csv'))        
        expected_posterior_samples    = pd.read_csv(os.path.join(expected_output_dir,
                                                    'posterior_samples.csv'))        
        
        # List all files in the tmp_path to verify the output files
        print("Files in the temporary directory:")
        for root, dirs, files in os.walk(tmp_path):
            for file in files:
                print(os.path.join(root, file))


        # Load the actual output files ------------------------------------------------------------------

        # Find the actual calibration_* directory, because the name is created dynamically with a timestamp
        calibration_dirs = glob.glob(os.path.join(
            tmp_path, 
            f'output_{n_samples}_{n_params}_params',
            'bayesian_calibration_results/17_output_keys',
            'calibration_*'
        ))

        assert len(calibration_dirs) == 1, "Expected exactly one calibration_* directory"
        calibration_dir = calibration_dirs[0]

        posterior_covariance = pd.read_csv(os.path.join(calibration_dir,
                                                        'posterior_covariance.csv'))
        posterior_mean       = pd.read_csv(os.path.join(calibration_dir,
                                                        'posterior_mean.csv'))
        posterior_samples    = pd.read_csv(os.path.join(calibration_dir,
                                                        'posterior_samples.csv'))                

        # Test equality ------------------------------------------------------------------------
        pd.testing.assert_frame_equal(expected_posterior_covariance, posterior_covariance)
        pd.testing.assert_frame_equal(expected_posterior_mean, posterior_mean)
        pd.testing.assert_frame_equal(expected_posterior_samples, posterior_samples)

