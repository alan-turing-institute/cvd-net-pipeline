import os
import numpy as np
import pandas as pd
from SALib import ProblemSpec
from multiprocessing import Pool
from functools import partial

def process_emulator(emulator_name,
                     pure_input_params,
                     emulators,
                     output_dir,
                     seed):
    """Process a single emulator for sensitivity analysis"""
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
    problem.sample_sobol(1024, seed=seed)
    X_samples = problem.samples
    
    # Extract emulator coefficients and intercept
    beta_matrix = np.array(linear_model.coef_)
    intercept = np.array(linear_model.intercept_)

    # Ensure correct shape
    if beta_matrix.ndim == 1:
        beta_matrix = beta_matrix.reshape(1, -1)
    if intercept.ndim == 0:
        intercept = np.array([intercept])
    
    # Compute emulator outputs
    Y_samples = X_samples @ beta_matrix.T + intercept
    Y_reshape = Y_samples.reshape(-1)

    # Set and analyze results
    problem.set_results(Y_reshape)
    sobol_indices = problem.analyze_sobol(print_to_console=False,
                                          seed=seed)

    # Sort results
    total, first, second = sobol_indices.to_df()
    total.sort_values('ST', inplace=True, ascending=False)

    # Save results
    result_data = pd.DataFrame({
        "Parameter": total.index,
        "ST": total['ST'],
        "ST_conf": total['ST_conf']
    })
    save_emulator_name = emulator_name.replace("/", "_")
    result_file = os.path.join(output_dir, f"sensitivity_{save_emulator_name}.csv")
    result_data.to_csv(result_file, index=False)
    
    return f"Completed {emulator_name}"    


def sensitivity_analysis(n_samples: int, 
                         n_params: int, 
                         output_path: str,
                         seed: int = 42,
                         n_processes: int = None,
                         output_dir_sims: str = None):

    file_suffix = output_dir_sims.split("output")[-1]

    # Read Input Data
    pure_input_params = pd.read_csv(f"{output_path}/pure_input{file_suffix}.csv")

    # Import Emulators
    emulators = pd.read_pickle(f"{output_path}/{output_dir_sims}/emulators/linear_models_and_r2_scores_{n_samples}.pkl")

    # Extract the emulator names
    emulator_list = emulators.index.to_list()

    # Directory to save results
    output_dir = f"{output_path}/{output_dir_sims}/sensitivity_analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    # Create partial function with fixed arguments
    process_func = partial(
        process_emulator,
        pure_input_params=pure_input_params,
        emulators=emulators,
        output_dir=output_dir,
        seed=seed
    )

    # Use multiprocessing
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_func, emulator_list)

    for result in results:
        print(result)   

    print("All analyses completed.")