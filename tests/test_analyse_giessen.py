import pytest
import pandas as pd
import os
import numpy as np
from src.analyse_giessen import analyse_giessen

@pytest.fixture
def temp_csv_file(tmp_path):
    # Create a temporary CSV file for testing
    file_path = tmp_path / "pressure_traces_rv"
    file_path.mkdir()
    csv_file = file_path / "all_pressure_traces.csv"
    headers = [str(i) for i in range(100)] + ["CO", "dt", "EF", "dPAP", "sPAP", "mPAP"]
    # Generate data
    data = []
    for _ in range(10):
        # Generate 100 numeric values (e.g., using a sine wave pattern for variety)
        numeric_values = np.sin(np.linspace(0, 10, 100)) * 30 + 30  # Example pattern
        # Append additional values
        additional_values = [
            np.random.uniform(3, 5),  # CO
            np.random.uniform(0.005, 0.01),  # dt
            np.random.uniform(0.3, 0.6),  # EF
            np.random.uniform(0.1, 0.2),  # dPAP
            np.random.uniform(25, 35),  # sPAP
            np.random.uniform(5, 10)  # mPAP
        ]
        data.append(list(numeric_values) + additional_values)

    # Create a DataFrame
    df = pd.DataFrame(data, columns=headers)
    pd.DataFrame(data, columns=headers).to_csv(csv_file, index=False)
    return tmp_path

def test_analyse_giessen_valid_input(temp_csv_file):
    # Call the function with the temporary file path
    analyse_giessen(temp_csv_file)

    # Check if the output file is created
    output_file = temp_csv_file / "waveform_resampled_all_pressure_traces_rv.csv"
    assert output_file.exists()

    # Validate the contents of the output file
    output_data = pd.read_csv(output_file)
    assert not output_data.empty

def test_analyse_giessen_invalid_input():
    # Test with an invalid file path
    with pytest.raises(FileNotFoundError):
        analyse_giessen("invalid/path")