import pytest
import pandas as pd
import os
import numpy as np
from analyse_giessen import analyse_giessen

@pytest.fixture
def cleanup_output_file():
    output_file = "tests/inputs_for_tests/analyse_giessen_module/output_64_9_params/waveform_resampled_all_pressure_traces_rv.csv"
    yield output_file
    if os.path.exists(output_file):
        os.remove(output_file)

@pytest.fixture
def cleanup_calibration_output_file():
    output_file = "tests/inputs_for_tests/analyse_giessen_module/output_64_9_params/bayesian_calibration_results/17_output_keys/calibration_20250604_154542/waveform_resampled_all_pressure_traces_rv.csv"
    yield output_file
    if os.path.exists(output_file):
        os.remove(output_file)        

def test_analyse_giessen_valid_input(cleanup_output_file):

    # Call the function with the input file
    analyse_giessen(file_path='tests/inputs_for_tests/analyse_giessen_module/output_64_9_params', 
                    data_type="synthetic",
                    gaussian_sigmas=[6., 4., 2.])

    # Check if the output data matches the expected output
    output_data = pd.read_csv(cleanup_output_file)
    expected_output = pd.read_csv('tests/expected_outputs/analyse_giessen_module/output_64_9_params/waveform_resampled_all_pressure_traces_rv.csv')
    pd.testing.assert_frame_equal(output_data[expected_output.columns], expected_output)

def test_analyse_giessen_invalid_input():
    # Test with an invalid file path
    with pytest.raises(FileNotFoundError):
        analyse_giessen(file_path="invalid/path",
                        data_type="synthetic",
                        gaussian_sigmas=[6., 4., 2.])

def test_analyse_giessen_valid_calibrated_input(cleanup_calibration_output_file):

    # Call the function with the input file
    analyse_giessen(file_path='tests/inputs_for_tests/analyse_giessen_module/output_64_9_params/bayesian_calibration_results/17_output_keys/calibration_20250604_154542/',
                    data_type="synthetic",
                     gaussian_sigmas=[6., 4., 2.])

    # Check if the output data matches the expected output
    output_data = pd.read_csv(cleanup_calibration_output_file)
    expected_output = pd.read_csv('tests/expected_outputs/analyse_giessen_module/output_64_9_params/bayesian_calibration_results/17_output_keys/calibration_20250604_154542/waveform_resampled_all_pressure_traces_rv.csv')
    pd.testing.assert_frame_equal(output_data[expected_output.columns], expected_output)
