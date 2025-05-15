import os
from ModularCirc.Models.KorakianitisMixedModel import KorakianitisMixedModel, KorakianitisMixedModel_parameters, TEMPLATE_TIME_SETUP_DICT
from ModularCirc import BatchRunner
import numpy as np
import pandas as pd
from utils import utils, plot_utils

def simulate_data(param_path: str, n_samples: int, output_path: str, repeat_simulations: bool = True):

    br = BatchRunner('Sobol', 0)
    br.setup_sampler(param_path)
    br.sample(n_samples)

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

    # count number of sampled parameters
    relevant_columns = []
    for col in br.samples.columns:
        relevant_columns.append(col)
        if col == 'T': break

    n_params = len(relevant_columns)
    br.map_vessel_volume()

    br.setup_model(model=KorakianitisMixedModel, po=KorakianitisMixedModel_parameters,
                   time_setup=TEMPLATE_TIME_SETUP_DICT)

    input_header = ','.join(br.samples.columns)

    np.savetxt(os.path.join(output_path,f'input_{n_samples}_{n_params}params.csv'), br.samples, header=input_header, delimiter=',')


    output_parameters = os.path.join(output_path, f'output_{n_samples}_{n_params}params')
    output_parameters_simulations = os.path.join(output_parameters,'simulations')

    # Check if the directory exists and contains n_samples files
    if os.path.exists(output_parameters_simulations) and len(os.listdir(output_parameters_simulations)) >= n_samples and repeat_simulations==False:
        print(f"Skipping simulation as {output_parameters} already contains 500 or more files.")
        # read a list of dataframes from here
        simulations = utils.load_simulation(output_parameters_simulations)

        if len(simulations) != n_samples:
            raise ValueError(f"Expected {n_samples} simulations, but found {len(simulations)}. Will run simulations again.")
            repeat_simulations = True
    else:
        print(f"Running simulation as {output_parameters}.")
        repeat_simulations = True


    if repeat_simulations:
        os.makedirs(output_parameters_simulations, exist_ok=True)
        simulations = br.run_batch(
            n_jobs=5,
            output_path=output_parameters_simulations
        )


    # Check for bool values in the list
    bool_indices = [index for index, value in enumerate(simulations) if isinstance(value, bool)]

    if bool_indices:
        print(f"Boolean values found at indices: {bool_indices}")
        print(f"Number of Booleans = {len(bool_indices)}")
    else:
        print("No boolean values found in the list.")


    utils.save_csv(pd.DataFrame(bool_indices), os.path.join(output_parameters, f'bool_indices_{n_samples}.csv'))

    # plot simulated traces
    plot_utils.plot_simulated_traces(simulations, output_path=output_parameters)

    # TODO always save the pressure traces despite of screening flag
    pressure_traces_df_pat, pressure_traces_df_rv = utils.select_feasible_traces(simulated_traces=simulations, screen=False, output_path=output_parameters)

    # Save the DataFrame to a single CSV file with headers
    utils.save_csv(pressure_traces_df_pat, f'{output_parameters}/pressure_traces_pat/all_pressure_traces.csv')
    utils.save_csv(pressure_traces_df_rv, f'{output_parameters}/pressure_traces_rv/all_pressure_traces.csv')

    plot_utils.plot_pressure_transients_arterial_tree(pressure_traces_df_rv, output_parameters)

    return None
