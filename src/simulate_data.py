import os
from ModularCirc.Models.KorakianitisMixedModel import KorakianitisMixedModel, KorakianitisMixedModel_parameters, TEMPLATE_TIME_SETUP_DICT
from ModularCirc import BatchRunner
import numpy as np
import pandas as pd
from utils import helper_functions, plot_utils

def simulate_data(param_path: str, n_samples: int, output_path: str, repeat_simulations: bool = True, sample_parameters: bool = True) -> str:
    
    # Sample parameters or read in from file
    
    br = BatchRunner('Sobol', 0)
    br.setup_sampler(param_path)
    br.sample(n_samples)

    if not sample_parameters:
        print("Simulating from calibrated parameters.")
        posterior_samples = pd.read_csv(f"{output_path}/posterior_samples.csv") 

        # remove any #s from column names
        posterior_samples.columns = posterior_samples.columns.str.lstrip('#').str.strip()

        for i, col in enumerate(posterior_samples.columns):
            br._samples.loc[:, col] = posterior_samples.loc[:,col]
            
    col_test = [col for col in br._samples.columns if col in br._parameters_2_sample.keys()]
    _pure_samples = br._samples[col_test].copy()

    
    map_ = {
            'delay': ['la.delay', 'ra.delay'],
            'td0': ['lv.td0', 'rv.td0'],
            'tr': ['lv.tr', 'rv.tr'],
            'tpww': ['la.tpww', 'ra.tpww'],
        }
        
    br.map_sample_timings(
        ref_time=1.,
        map=map_
    )

    br._samples[['lv.td', 'rv.td']] = br._samples[['lv.tr', 'rv.tr']].values + br._samples[['lv.td0', 'rv.td0']].values
    br._samples.drop(['lv.td0', 'rv.td0'], axis=1, inplace=True)

    # get relevant columns and count them
    relevant_columns = _pure_samples.columns.to_list()

    n_params = len(relevant_columns)
    br.map_vessel_volume()

    br.setup_model(model=KorakianitisMixedModel, po=KorakianitisMixedModel_parameters,
                   time_setup=TEMPLATE_TIME_SETUP_DICT)

    input_header = ','.join(br.samples.columns)
    file_suffix  = f'_{n_samples}_{n_params}_params'

    np.savetxt(os.path.join(output_path,f'pure_input{file_suffix}.csv'), _pure_samples, header=','.join(_pure_samples.columns), delimiter=',')
    np.savetxt(os.path.join(output_path,f'input{file_suffix}.csv'), br.samples, header=input_header, delimiter=',')

    if sample_parameters:
        output_parameters = os.path.join(output_path, f'output{file_suffix}')
        output_parameters_simulations = os.path.join(output_parameters,'simulations')
        
    else:
        output_parameters = output_path
        output_parameters_simulations = os.path.join(output_parameters, f'posterior_simulations')

    # Check if the directory exists and contains n_samples files
    if os.path.exists(output_parameters_simulations) and len(os.listdir(output_parameters_simulations)) >= n_samples and repeat_simulations==False:
        print(f"Skipping simulation as {output_parameters} already contains {n_samples} or more files.")
        # read a list of dataframes from here
        simulations = helper_functions.load_simulation(output_parameters_simulations)

        if len(simulations) != n_samples:
            raise ValueError(f"Expected {n_samples} simulations, but found {len(simulations)}. Will run simulations again.")
            repeat_simulations = True
    else:
        print(f"Running simulation as {output_parameters}.")
        repeat_simulations = True


    if repeat_simulations:
        os.makedirs(output_parameters_simulations, exist_ok=True)
        simulations = br.run_batch(
            n_jobs=10,
            output_path=output_parameters_simulations
        )


    # Check for bool values in the list
    bool_indices = [index for index, value in enumerate(simulations) if isinstance(value, bool)]

    if bool_indices:
        print(f"Boolean values found at indices: {bool_indices}")
        print(f"Number of Booleans = {len(bool_indices)}")
    else:
        print("No boolean values found in the list.")


    helper_functions.save_csv(pd.DataFrame(bool_indices), os.path.join(output_parameters, f'bool_indices_{n_samples}.csv'))

    # plot simulated traces
    plot_utils.plot_simulated_traces(simulations, output_path=output_parameters)

    pressure_traces_df_pat, pressure_traces_df_rv = helper_functions.select_feasible_traces(simulated_traces=simulations, output_path=output_parameters)

    # Save the DataFrame to a single CSV file with headers
    helper_functions.save_csv(pressure_traces_df_pat, f'{output_parameters}/pressure_traces_pat/all_pressure_traces.csv')
    helper_functions.save_csv(pressure_traces_df_rv, f'{output_parameters}/pressure_traces_rv/all_pressure_traces.csv')

    plot_utils.plot_pressure_transients_arterial_tree(pressure_traces_df_rv, output_parameters)

    return output_parameters, n_params
