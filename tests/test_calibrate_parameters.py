import os
import pytest
import pandas as pd
from calibrate_parameters import calibrate_parameters
import tempfile
import shutil
import glob

def test_calibrate_parameters():
    # Define test parameters

    with tempfile.TemporaryDirectory() as tmp_path:
        print(f"Temporary directory created at: {tmp_path}")

        output_keys = [
            "t_max_dpdt", "a_epad", "epad", "s_a_epad", "s_epad",
            "min_dpdt", "max_dpdt", "A_p", "P_max", "esp", "sys",
            "EF", "Ees/Ea", "iT", "PC1", "PC2", "PC3"
        ]

        n_samples = 64
        n_params = 9

        # Copy all the expected input files from /tests/inputs_for_tests/calibrate_parameters_module/ to the temporary directory
        shutil.copytree('./tests/inputs_for_tests/calibrate_parameters_module',
                        tmp_path,
                        dirs_exist_ok=True)

        calibrate_parameters(data_type="synthetic",
                             n_samples=n_samples,
                             n_params=n_params,
                             output_path=str(tmp_path),
                             output_keys=output_keys,
                             include_timeseries=False,
                             epsilon_obs_scale=0.05,
                             config=[])

        # Compare the output files to the expected output files
        expected_output_dir = './tests/expected_outputs/calibrate_parameters_module/output_64_9_params/bayesian_calibration_results/17_output_keys/calibration_20250604_100806'
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


        # Load the actual output files

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

        pd.testing.assert_frame_equal(expected_posterior_covariance, posterior_covariance)
        pd.testing.assert_frame_equal(expected_posterior_mean, posterior_mean)
        pd.testing.assert_frame_equal(expected_posterior_samples, posterior_samples)

