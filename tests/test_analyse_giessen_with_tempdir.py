"""
Test module for analyse_giessen function using temporary directories.

This test file is an improved version of test_analyse_giessen.py that:
1. Uses tempfile.TemporaryDirectory() for clean test isolation
2. Copies input data to temporary directories to avoid polluting source data
3. Compares results against known good outputs in tests/known_good_outputs/
4. Follows the same pattern as test_simulate_data.py for consistency
5. Properly handles the function's behavior of writing output to the input directory

Key improvements over the original:
- No manual cleanup fixtures needed - temporary directories are auto-cleaned
- No risk of test interference from leftover files
- Better test isolation and repeatability
- More robust error handling and edge case testing
"""

import pytest
import pandas as pd
import os
import tempfile
import shutil
from cvdnet_pipeline.analyse_giessen import analyse_giessen
from test_constants import (
    DEFAULT_N_SAMPLES,
    DEFAULT_N_PARAMS,
    DEFAULT_EPSILON_OBS_SCALE,
)

RTOL_TOLERANCE = 1e-2

# @pytest.mark.skip(reason="Temporarily disabled")
def test_analyse_giessen_synthetic_data():
    """Test analyse_giessen with synthetic data using temporary directories."""
    
    with tempfile.TemporaryDirectory() as tmp_path:
        print(f"Temporary directory created at: {tmp_path}")
        
        # Define input and expected output paths
        input_data_path      = ('./tests/known_good_outputs/synthetic_data/'
                                f'output_{DEFAULT_N_SAMPLES}_{DEFAULT_N_PARAMS}_params/')
        expected_output_path = os.path.join(input_data_path,'waveform_resampled_all_pressure_traces_rv.csv')
        
        # Verify input data exists
        assert os.path.exists(input_data_path), f"Input data path does not exist: {input_data_path}"
        assert os.path.exists(expected_output_path), f"Expected output path does not exist: {expected_output_path}"
        
        # Copy input data to temporary directory for processing
        shutil.copytree(input_data_path, tmp_path, dirs_exist_ok=True)
        
        # The output file will be created in the same directory as the input
        output_file = os.path.join(tmp_path, 'waveform_resampled_all_pressure_traces_rv.csv')
        
        # Call analyse_giessen function
        analyse_giessen(
            file_path=tmp_path,
            data_type="synthetic",
            gaussian_sigmas=[1e-05, 
                             1e-05,
                             1e-05]
        )
        
        # Verify output file was created
        assert os.path.exists(output_file), f"Output file was not created: {output_file}"
        
        # Load and compare results
        output_data = pd.read_csv(output_file)
        expected_output = pd.read_csv(expected_output_path)
        
        # Compare the DataFrames
        pd.testing.assert_frame_equal(
            output_data[expected_output.columns], 
            expected_output,
            check_exact=False,
            rtol=RTOL_TOLERANCE
        )


# @pytest.mark.skip(reason="Temporarily disabled")
def test_analyse_giessen_real_data():
    """Test analyse_giessen with real data using temporary directories."""
    
    with tempfile.TemporaryDirectory() as tmp_path:
        print(f"Temporary directory created at: {tmp_path}")
        
        # Define input and expected output paths
        input_data_path = f'./tests/known_good_outputs/real_data/'
        expected_output_path = os.path.join(input_data_path, 'waveform_resampled_all_pressure_traces_rv.csv')
        
        # Check if real data exists (may not be available in all test environments)
        if not os.path.exists(input_data_path):
            pytest.skip("Real data not available for testing")
        
        # Copy input data to temporary directory for processing
        shutil.copytree(input_data_path, tmp_path, dirs_exist_ok=True)
        
        # The output file will be created in the same directory as the input
        output_file = os.path.join(tmp_path, 'waveform_resampled_all_pressure_traces_rv.csv')
        
        # Call analyse_giessen function
        analyse_giessen(
            file_path=tmp_path,
            data_type="real",
            gaussian_sigmas=[6., 4., 2.]
        )
        
        # Verify output file was created
        assert os.path.exists(output_file), f"Output file was not created: {output_file}"
        
        # Load and compare results
        output_data = pd.read_csv(output_file)
        expected_output = pd.read_csv(expected_output_path)
        
        # Compare the DataFrames
        pd.testing.assert_frame_equal(
            output_data[expected_output.columns], 
            expected_output,
            check_exact=False,
            rtol=RTOL_TOLERANCE
        )


# @pytest.mark.skip(reason="Temporarily disabled")
def test_analyse_giessen_calibrated_synthetic_data():
    """Test analyse_giessen with calibrated synthetic data using temporary directories."""
    
    with tempfile.TemporaryDirectory() as tmp_path:
        print(f"Temporary directory created at: {tmp_path}")
        
        # Define input and expected output paths for calibrated data
        calibration_subdir = 'bayesian_calibration_results/17_output_keys/calibration_20250917_135136'
        input_data_path = f'./tests/known_good_outputs/synthetic_data/output_{DEFAULT_N_SAMPLES}_{DEFAULT_N_PARAMS}_params/{calibration_subdir}/'
        expected_output_path = f'./tests/known_good_outputs/synthetic_data/output_{DEFAULT_N_SAMPLES}_{DEFAULT_N_PARAMS}_params/{calibration_subdir}/waveform_resampled_all_pressure_traces_rv.csv'
        
        # Check if calibrated data exists
        if not os.path.exists(input_data_path):
            pytest.skip("Calibrated synthetic data not available for testing")
        
        # Copy input data to temporary directory for processing
        shutil.copytree(input_data_path, tmp_path, dirs_exist_ok=True)
        
        # The output file will be created in the same directory as the input
        output_file = os.path.join(tmp_path, 'waveform_resampled_all_pressure_traces_rv.csv')
        
        # Call analyse_giessen function
        analyse_giessen(
            file_path=tmp_path,
            data_type="synthetic",
            gaussian_sigmas=[1e-05, 
                             1e-05,
                             1e-05]
        )
        
        # Verify output file was created
        assert os.path.exists(output_file), f"Output file was not created: {output_file}"
        
        # Load and compare results
        output_data = pd.read_csv(output_file)
        expected_output = pd.read_csv(expected_output_path)
        
        # Compare the DataFrames
        pd.testing.assert_frame_equal(
            output_data[expected_output.columns], 
            expected_output,
            check_exact=False,
            rtol=RTOL_TOLERANCE
        )

# @pytest.mark.skip(reason="Temporarily disabled")
def test_analyse_giessen_invalid_input():
    """Test analyse_giessen with invalid file path."""
    
    with tempfile.TemporaryDirectory() as tmp_path:
        # Test with an invalid file path
        with pytest.raises(FileNotFoundError):
            analyse_giessen(
                file_path="invalid/path",
                data_type="synthetic",
                gaussian_sigmas=[6., 4., 2.]
            )


# @pytest.mark.skip(reason="Temporarily disabled")
def test_analyse_giessen_missing_required_files():
    """Test analyse_giessen behavior when required input files are missing."""
    
    with tempfile.TemporaryDirectory() as tmp_path:
        # Create a directory without the required input files
        empty_input_path = os.path.join(tmp_path, 'empty_input')
        os.makedirs(empty_input_path)
        
        # Test with empty directory - should raise an error
        with pytest.raises((FileNotFoundError, KeyError, pd.errors.EmptyDataError)):
            analyse_giessen(
                file_path=empty_input_path,
                data_type="synthetic",
                gaussian_sigmas=[6., 4., 2.]
            )