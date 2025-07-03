import pytest
import pandas as pd
import os
import numpy as np
from build_emulator import build_emulator
import shutil

@pytest.fixture
def cleanup_output():
    output_file = "tests/inputs_for_tests/build_emulator_module/output_50_9_params/emulators/linear_models_and_r2_scores_50.csv"

    yield output_file
    if os.path.exists(output_file):
        shutil.rmtree('tests/inputs_for_tests/build_emulator_module/output_50_9_params/emulators/')

def test_build_emulator(cleanup_output):


    output_keys = [
        "t_max_dpdt",
        "a_epad",
        "epad",
        "s_a_epad",
        "s_epad",
        "A_p",
        "P_max",
        "esp",
        "sys",
        "iT",
        "PC1",
        "PC2",
        "PC3"
        ]

    # Call the function with the input file
    build_emulator(n_samples=50, 
                   n_params=9,
                   output_path='tests/inputs_for_tests/build_emulator_module',
                   output_file_name="waveform_resampled_all_pressure_traces_rv_with_pca.csv",
                   output_keys_red=output_keys)

    # Check if the output data matches the expected output
    output_data = pd.read_csv(cleanup_output)
    expected_output = pd.read_csv('tests/expected_outputs/build_emulator_module/output_50_9_params/emulators/linear_models_and_r2_scores_50.csv')
    pd.testing.assert_frame_equal(output_data, expected_output)