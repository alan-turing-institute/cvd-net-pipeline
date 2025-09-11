import os
import pytest
import pandas as pd
from calibrate_parameters import calibrate_parameters
import tempfile
import shutil
import glob
from tests.test_constants import (
    OUTPUT_KEYS_FOR_TESTS_FOR_REAL_DATA_CALIBRATION, 
    DEFAULT_N_SAMPLES, 
    DEFAULT_N_PARAMS, 
    DEFAULT_EPSILON_OBS_SCALE
)


def test_calibrate_parameters_real():
    # Define test parameters

    with tempfile.TemporaryDirectory() as tmp_path:
        print(f"Temporary directory created at: {tmp_path}")

        output_keys = OUTPUT_KEYS_FOR_TESTS_FOR_REAL_DATA_CALIBRATION
        n_samples = DEFAULT_N_SAMPLES
        n_params = DEFAULT_N_PARAMS

        # Copy all the expected input files to the temporary directory
        emulator_path = './tests/known_good_outputs/synthetic_data/'
        # From emulator_path, copy the input file: pure_input_64_9_params.csv
        shutil.copy(os.path.join(emulator_path, f'pure_input_{n_samples}_{n_params}_params.csv'), tmp_path)
        # Now copy over the waveform data:
        shutil.copy('./tests/known_good_outputs/real_data/waveform_resampled_all_pressure_traces_rv_with_pca.csv', tmp_path)

        calibrate_parameters(data_type="real",
                             n_samples=n_samples,
                             n_params=n_params,
                             output_path=str(tmp_path),
                             emulator_path=emulator_path,
                             output_keys=output_keys,
                             include_timeseries=False,
                             epsilon_obs_scale=DEFAULT_EPSILON_OBS_SCALE,
                             config=[])

        # Compare the output files to the expected output files

        # Load the expected output ----------------------------------------------------------------------
        expected_output_dir = os.path.join(
            './tests/known_good_outputs/real_data/',
            'bayesian_calibration_results',
            '15_output_keys',
            'calibration_20250911_133944'
        )
        
        expected_posterior_covariance = pd.read_csv(os.path.join(expected_output_dir,
                                                    'posterior_covariance.csv'))
        expected_posterior_mean       = pd.read_csv(os.path.join(expected_output_dir,
                                                    'posterior_mean.csv'))                
        
        # List all files in the tmp_path to verify the output files
        print("Files in the temporary directory:")
        for root, dirs, files in os.walk(tmp_path):
            for file in files:
                print(os.path.join(root, file))


        # Load the actual output files ------------------------------------------------------------------

        # Find the actual calibration_* directory, because the name is created dynamically with a timestamp
        calibration_dirs = glob.glob(os.path.join(
            tmp_path,
            'bayesian_calibration_results/15_output_keys',
            'calibration_*'
        ))

        print(f"Found calibration directories: {calibration_dirs}")
        assert len(calibration_dirs) == 1, "Expected exactly one calibration_* directory"
        calibration_dir = calibration_dirs[0]

        posterior_covariance = pd.read_csv(os.path.join(calibration_dir,
                                                        'posterior_covariance.csv'))
        posterior_mean       = pd.read_csv(os.path.join(calibration_dir,
                                                        'posterior_mean.csv'))              

        # Test rough equality ------------------------------------------------------------------------
        pd.testing.assert_frame_equal(
            expected_posterior_covariance, 
            posterior_covariance, 
            atol=1e-4, 
            rtol=1e-4, 
            check_exact=False
        )
        pd.testing.assert_frame_equal(
            expected_posterior_mean, 
            posterior_mean, 
            atol=1e-4, 
            rtol=1e-4, 
            check_exact=False
        )

