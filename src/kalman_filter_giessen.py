import numpy as np
import pandas as pd

def KFGiessenSETUP(n_samples:int=4096, 
                n_params:int=9, 
                output_path:str='output', 
                emulator_path:str=None,
                output_keys:list=None,
                include_timeseries:bool=True,
                epsilon_obs_scale:float=0.05,
                ):
        
    # Load observation data
    output_file = pd.read_csv(f"{output_path}/waveform_resampled_all_pressure_traces_rv_with_pca.csv")

    # Input for priors
    input_prior = pd.read_csv(f'{emulator_path}/input_{n_samples}_{n_params}_params.csv')

    # emulators
    emulators = pd.read_pickle(f"{emulator_path}/output_{n_samples}_{n_params}_params/emulators/linear_models_and_r2_scores_{n_samples}.pkl")

    if include_timeseries:
        all_output_keys = output_file.iloc[:, :101].columns.tolist() + output_keys
        print("Including time-series in calibraiton as specified in config file.")

        # Build the diagonal entries: 101 ones followed by the std devs
        # 101 ones are scaled by epsilon_obs_scale so they will equal 
        # 1 when multipled by epsilon_obs_scale further down. 
        sd_values = output_file[output_keys].std().values
        diagonal_values = np.concatenate([np.ones(101)/epsilon_obs_scale, sd_values]) 
    else:
        all_output_keys = output_keys
        sd_values = output_file[output_keys].std().values
        diagonal_values = sd_values

    # Create the diagonal matrix
    e_obs = np.diag(diagonal_values) * epsilon_obs_scale
    
    # Select emulators and data for specified output_keys
    emulator_output = emulators.loc[all_output_keys]
    observation_data = output_file.loc[:, all_output_keys]

    # Priors
    mu_0 = np.array(input_prior.mean().loc[:'T'])
    mu_0 = mu_0.reshape(-1, 1)
    Sigma_0 = np.diag(input_prior.var().loc[:'T'])

    # dynamically define prior on T
    mu_0[-1,-1] = observation_data['iT'].iloc[0]
    Sigma_0[-1, -1] = 0.0000001

    # Parameter names
    param_names = input_prior.loc[:, :'T'].columns.to_list()

    # Model error
    epsilon_model = np.diag(emulator_output['RSE']**2) 

    # Construct beta matrix and intercepts
    beta_matrix = []
    intercept = []

    for _, row_entry in emulator_output.iterrows():
        model = row_entry['Model']
        beta_matrix.append(model.coef_)
        intercept.append(model.intercept_)
    
    beta_matrix = np.array(beta_matrix)
    intercept = np.array(intercept).reshape(len(intercept), 1)
    
    return beta_matrix, intercept, e_obs, epsilon_model, mu_0, Sigma_0, param_names, observation_data