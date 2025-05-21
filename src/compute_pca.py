import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import plot_utils


def compute_pca(n_samples:int=500, n_params:int=5, n_pca_components:int=10, output_path:str="output"):
    print("[ComputePCA] Running PCA (placeholder)")

    output_file_name = 'waveform_resampled_all_pressure_traces_rv.csv'

    output_file = pd.read_csv(f"{output_path}/output_{n_samples}_{n_params}params/{output_file_name}")

    # Create direcoty for results
    if not os.path.exists(f"{output_path}/output_{n_samples}_{n_params}params/pca"):
        os.makedirs(f"{output_path}/output_{n_samples}_{n_params}params/pca")

    bool_exist = False
    if bool_exist:
        boolean_index = pd.read_csv(f"{output_path}/output_{n_samples}_{n_params}params/bool_indices_{n_samples}.csv")

    ## Conduct PCA ##
    df = output_file.copy()

    # Copy the data and separate the target variable (only pressure traces)
    X = df.iloc[:,:100].copy() # traces only

    # Create an instance of StandardScaler
    scaler = StandardScaler()

    # Fit the scaler to the data and transform it - standardize
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_pca_components)
    X_pca = pca.fit_transform(X_scaled)

    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names, index=df.index)

    X_pca.to_csv(f'{output_path}/output_{n_samples}_{n_params}params/pca/PCA.csv', index=False)

    # Concatenate the PCA components with the original data     
    df_pca = pd.concat([df, X_pca], axis=1)

    # Create a new name for the output file which appends "_with_pca" to the original name
    output_file_name_pca = output_file_name.replace('.csv', '_with_pca.csv')

    df_pca.to_csv(f'{output_path}/output_{n_samples}_{n_params}params/{output_file_name_pca}', index=False)

    output_parameters = os.path.join(output_path, f'output_{n_samples}_{n_params}params')

    # Plot the PCA histogram
    plot_utils.plot_pca_histogram(X_pca, output_path=output_parameters, n_pca_components=n_pca_components)

    # Plot the explained variance ratio
    plot_utils.plot_pca_explained_variance(pca, output_path=output_parameters)

    # Plot the PCA transformed data
    plot_utils.plot_pca_transformed(pca, X_scaled, output_path=output_parameters)

