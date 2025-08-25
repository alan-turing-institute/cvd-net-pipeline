import pytest
import pandas as pd
import os
from analyse_giessen import analyse_giessen


# Parametrize data types for testing both synthetic and real data
DATA_TYPES = ["real"]


@pytest.fixture(params=DATA_TYPES)
def data_type(request):
    """Fixture that provides both synthetic and real data types for parameterized tests."""
    return request.param


@pytest.fixture
def cleanup_output_file(data_type):
    """Cleanup fixture that handles output files for both data types."""
    output_file = f"tests/inputs_for_tests/analyse_giessen_module/output_64_9_params/{data_type}_data/waveform_resampled_all_pressure_traces_rv.csv"
    yield output_file, data_type
    if os.path.exists(output_file):
        os.remove(output_file)

@pytest.fixture
def cleanup_calibration_output_file():
    """Cleanup fixture for calibrated synthetic data (only available for synthetic data currently)."""
    output_file = "tests/inputs_for_tests/analyse_giessen_module/output_64_9_params/synthetic_data/bayesian_calibration_results/17_output_keys/calibration_20250604_154542/waveform_resampled_all_pressure_traces_rv.csv"
    yield output_file
    if os.path.exists(output_file):
        os.remove(output_file)        

def test_analyse_giessen_valid_input(cleanup_output_file):
    """Test analyse_giessen with valid input for both synthetic and real data."""
    output_file, data_type = cleanup_output_file
    
    # Call the function with the input file
    analyse_giessen(file_path=f'tests/inputs_for_tests/analyse_giessen_module/output_64_9_params/{data_type}_data/', 
                    data_type=data_type,
                    gaussian_sigmas=[6., 4., 2.])

    # Check if the output data matches the expected output
    output_data = pd.read_csv(output_file)
    expected_output = pd.read_csv(f'tests/expected_outputs/analyse_giessen_module/output_64_9_params/{data_type}_data/waveform_resampled_all_pressure_traces_rv.csv')
    pd.testing.assert_frame_equal(output_data[expected_output.columns], expected_output)


@pytest.mark.parametrize("data_type", DATA_TYPES)
def test_analyse_giessen_invalid_input(data_type):
    """Test analyse_giessen with invalid file path for both data types."""
    # Test with an invalid file path
    with pytest.raises(FileNotFoundError):
        analyse_giessen(file_path="invalid/path",
                        data_type=data_type,
                        gaussian_sigmas=[6., 4., 2.])

def test_analyse_giessen_valid_calibrated_input(cleanup_calibration_output_file):
    """Test analyse_giessen with valid calibrated input (synthetic data only currently)."""

    # Call the function with the input file
    analyse_giessen(file_path='tests/inputs_for_tests/analyse_giessen_module/output_64_9_params/synthetic_data/bayesian_calibration_results/17_output_keys/calibration_20250604_154542/',
                    data_type="synthetic",
                     gaussian_sigmas=[6., 4., 2.])

    # Check if the output data matches the expected output
    output_data = pd.read_csv(cleanup_calibration_output_file)
    expected_output = pd.read_csv('tests/expected_outputs/analyse_giessen_module/output_64_9_params/synthetic_data/bayesian_calibration_results/17_output_keys/calibration_20250604_154542/waveform_resampled_all_pressure_traces_rv.csv')
    pd.testing.assert_frame_equal(output_data[expected_output.columns], expected_output)
