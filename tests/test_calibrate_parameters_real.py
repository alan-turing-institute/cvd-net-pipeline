import os
import pytest
import pandas as pd
from calibrate_parameters import calibrate_parameters
import tempfile
import shutil
import glob

def test_calibrate_parameters_real():
    # Define test parameters

    with tempfile.TemporaryDirectory() as tmp_path:
        print(f"Temporary directory created at: {tmp_path}")

        output_keys = [
            "t_max_dpdt", 
            "a_epad",
            "epad", 
            "s_a_epad", 
            "s_epad",
            "min_dpdt", 
            "max_dpdt", 
            "A_p", 
            "P_max", 
            "esp", 
            "sys",
            "iT", 
            "PC1", 
            "PC2", 
            "PC3"
        ]

        n_samples = 64
        n_params = 9

        # Copy all the expected input files from /tests/inputs_for_tests/calibrate_parameters_module/ to the temporary directory
        shutil.copytree('./tests/inputs_for_tests/calibrate_parameters_module/real_data',
                        tmp_path,
                        dirs_exist_ok=True)

        calibrate_parameters(data_type="real",
                             n_samples=n_samples,
                             n_params=n_params,
                             output_path=str(tmp_path),
                             emulator_path='./tests/inputs_for_tests/calibrate_parameters_module/real_data',
                             output_keys=output_keys,
                             include_timeseries=False,
                             epsilon_obs_scale=0.05,
                             dummy_data_dir='./tests/inputs_for_tests/calibrate_parameters_module/dummy_data/',
                             config=[])

        # Compare the output files to the expected output files

        # Load the expected output ----------------------------------------------------------------------
        expected_output_dir = os.path.join(
            './tests/expected_outputs/calibrate_parameters_module',
            'real_data',
            'bayesian_calibration_results',
            '15_output_keys',
            'calibration_20250827_220413'
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

