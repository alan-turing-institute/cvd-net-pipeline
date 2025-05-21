import pytest
import pandas as pd
import os
import numpy as np
from analyse_giessen import analyse_giessen

@pytest.fixture
def cleanup_output_file():
    output_file = "tests/inputs_for_tests/analyse_giessen_module/output_10_9params/waveform_resampled_all_pressure_traces_rv.csv"
    yield output_file
    if os.path.exists(output_file):
        os.remove(output_file)

def test_analyse_giessen_valid_input(cleanup_output_file):

    # Call the function with the input file
    analyse_giessen('tests/inputs_for_tests/analyse_giessen_module/output_10_9params')

    # Check if the output data matches the expected output
    output_data = pd.read_csv(cleanup_output_file)
    expected_output = pd.read_csv('tests/expected_outputs/analyse_giessen_module/output_10_9params/waveform_resampled_all_pressure_traces_rv.csv')
    pd.testing.assert_frame_equal(output_data, expected_output)

def test_analyse_giessen_invalid_input():
    # Test with an invalid file path
    with pytest.raises(FileNotFoundError):
        analyse_giessen("invalid/path")