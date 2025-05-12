import os
from ModularCirc.Models.KorakianitisMixedModel import KorakianitisMixedModel, KorakianitisMixedModel_parameters, TEMPLATE_TIME_SETUP_DICT
from ModularCirc import BatchRunner
import numpy as np
import pandas as pd

def simulate_data(param_path: str, n_sample: int, input_path: str, output_path: str):

    br = BatchRunner('Sobol', 0)
    br.setup_sampler(param_path)
    br.sample(n_sample)

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

    np.savetxt(f'output/input_{n_sample}_{n_params}params.csv', br.samples, header=input_header, delimiter=',')

    os.system(f'mkdir -p output/output_{n_sample}_{n_params}params/output_{n_sample}')
    test = br.run_batch(n_jobs=5,
                        output_path=f'output/output_{n_sample}_{n_params}params/output_{n_sample}')

    # Check for bool values in the list
    bool_indices = [index for index, value in enumerate(test) if isinstance(value, bool)]

    if bool_indices:
        print(f"Boolean values found at indices: {bool_indices}")
        print(f"Number of Booleans = {len(bool_indices)}")
    else:
        print("No boolean values found in the list.")

    bool_indices_df = pd.DataFrame(bool_indices)
    bool_indices_df.to_csv(f"output/output_{n_sample}_{n_params}params/bool_indices_{n_sample}.csv", index=False)

    os.system(f'mkdir -p output/output_{n_sample}_{n_params}params/pressure_traces_pat')
    os.system(f'mkdir -p output/output_{n_sample}_{n_params}params/pressure_traces_rv')

    return input_path, output_path
