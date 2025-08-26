import pytest
import pandas as pd
import os
import numpy as np
from compute_pca import compute_pca
import shutil


# Parametrize data types for testing both synthetic and real data
DATA_TYPES = ["synthetic", "real"]


@pytest.fixture(params=DATA_TYPES)
def data_type(request):
    """Fixture that provides both synthetic and real data types for parameterized tests."""
    return request.param


@pytest.fixture
def cleanup_output(data_type):
    """Cleanup fixture that handles output files for both data types."""
    if data_type == "synthetic":
        output_file           = f"tests/inputs_for_tests/compute_pca_module/output_64_9_params/{data_type}_data/output_64_9_params/waveform_resampled_all_pressure_traces_rv_with_pca.csv"
        output_pca_folder     = f"tests/inputs_for_tests/compute_pca_module/output_64_9_params/{data_type}_data/output_64_9_params/pca"
        output_figures_folder = f"tests/inputs_for_tests/compute_pca_module/output_64_9_params/{data_type}_data/output_64_9_params/figures"

    elif data_type == "real":
        output_file           = f"tests/inputs_for_tests/compute_pca_module/output_64_9_params/{data_type}_data/waveform_resampled_all_pressure_traces_rv_with_pca.csv"
        output_pca_folder     = f"tests/inputs_for_tests/compute_pca_module/output_64_9_params/{data_type}_data/pca"
        output_figures_folder = f"tests/inputs_for_tests/compute_pca_module/output_64_9_params/{data_type}_data/figures"


    yield output_file, data_type
    
    # Cleanup files and directories
    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(output_pca_folder):
        shutil.rmtree(output_pca_folder)
    if os.path.exists(output_figures_folder):
        shutil.rmtree(output_figures_folder)                


def test_compute_pca(cleanup_output):
    """Test compute_pca with valid input for both synthetic and real data."""
    output_file, data_type = cleanup_output

    # Call the function with the input file
    compute_pca(n_samples=64, 
                n_params=9,
                n_pca_components=10,
                output_path=f'tests/inputs_for_tests/compute_pca_module/output_64_9_params/{data_type}_data',
                data_type=data_type)

    # Check if the output data matches the expected output
    output_data = pd.read_csv(output_file)
    expected_output = pd.read_csv(f'tests/expected_outputs/compute_pca_module/output_64_9_params/{data_type}_data/waveform_resampled_all_pressure_traces_rv_with_pca.csv')
    pd.testing.assert_frame_equal(output_data, expected_output)