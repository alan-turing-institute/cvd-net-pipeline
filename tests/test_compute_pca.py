import pytest
import pandas as pd
import os
import numpy as np
from compute_pca import compute_pca
import shutil

@pytest.fixture
def cleanup_output():
    output_file = "tests/inputs_for_tests/compute_pca_module/output_64_9_params/waveform_resampled_all_pressure_traces_rv_with_pca.csv"
    output_pca_folder = "tests/inputs_for_tests/compute_pca_module/output_64_9_params/pca"
    output_figures_folder = "tests/inputs_for_tests/compute_pca_module/output_64_9_params/figures"
    yield output_file
    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(output_pca_folder):
        shutil.rmtree(output_pca_folder)
    if os.path.exists(output_figures_folder):
        shutil.rmtree(output_figures_folder)                

def test_compute_pca(cleanup_output):

    # Call the function with the input file
    compute_pca(n_samples=64, 
                n_params=9,
                n_pca_components=10,
                output_path='tests/inputs_for_tests/compute_pca_module',
                data_type="synthetic")

    # Check if the output data matches the expected output
    output_data = pd.read_csv(cleanup_output)
    expected_output = pd.read_csv('tests/expected_outputs/compute_pca_module/output_64_9_params/waveform_resampled_all_pressure_traces_rv_with_pca.csv')
    pd.testing.assert_frame_equal(output_data, expected_output)