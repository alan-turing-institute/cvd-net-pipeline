import os
import numpy as np
import pandas as pd
from SALib import ProblemSpec

def sensitivity_analysis(n_samples: int, 
                         n_params: int, 
                         output_path: str):

    file_suffix = f'_{n_samples}_{n_params}_params'

    # Read Input Data
    pure_input_params = pd.read_csv(f"{output_path}/pure_input{file_suffix}.csv")s]

    # Import Emulators
    emulators = pd.read_pickle(f"{output_path}/output{file_suffix}/emulators/linear_models_and_r2_scores_{n_samples}.pkl")

    # Extract the emulator names
    emulator_list = emulators.index.to_list()

    # Directory to save results
    output_dir = f"{output_path}/output{file_suffix}/sensitivity_analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each emulator
    for emulator_name in emulator_list:
        print(f"Processing {emulator_name}...")

        # Define problem spec for sensitivity analysis
        problem = ProblemSpec({
            'num_vars': len(pure_input_params.columns),
            'names': pure_input_params.columns.tolist(),
            'bounds': pure_input_params.describe().loc[['min', 'max']].T.values,
            "outputs": [emulator_name],
        })

        linear_model = emulators.loc[emulator_name, 'Model']

        # Sample inputs
        problem.sample_sobol(1024)
        X_samples = problem.samples
        
        # Extract emulator coefficients and intercept
        beta_matrix = np.array(linear_model.coef_)  # (num_outputs, num_vars) or (num_vars,)
        intercept = np.array(linear_model.intercept_)  # (num_outputs,) or scalar

        # Ensure correct shape
        if beta_matrix.ndim == 1:
            beta_matrix = beta_matrix.reshape(1, -1)  # Convert (num_vars,) → (1, num_vars)
        if intercept.ndim == 0:
            intercept = np.array([intercept])  # Convert scalar → (1,)
        
        # Compute emulator outputs
        Y_samples = X_samples @ beta_matrix.T + intercept  # (num_samples, num_outputs)

        # Flatten if necessary
        Y_reshape = Y_samples.reshape(-1)
        #print(f"Y_reshape shape: {Y_reshape.shape}")

        # Set and analyze results
        problem.set_results(Y_reshape)
        sobol_indices = problem.analyze_sobol(print_to_console=False)

        # Sort results
        total, first, second = sobol_indices.to_df()
        total.sort_values('ST', inplace=True, ascending=False)

        # Save results
        result_data = pd.DataFrame({
            "Parameter": total.index,
            "ST": total['ST'],
            "ST_conf": total['ST_conf']
        })
        save_emulator_name = emulator_name.replace("/", "_") # Avoid slashes in file names

        result_file = os.path.join(output_dir, f"sensitivity_{save_emulator_name}.csv")
        result_data.to_csv(result_file, index=False)

    print("All analyses completed.")