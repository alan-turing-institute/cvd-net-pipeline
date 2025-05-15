from ModularCirc.Models.KorakianitisMixedModel import KorakianitisMixedModel, KorakianitisMixedModel_parameters, TEMPLATE_TIME_SETUP_DICT
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  PowerTransformer
import contextlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from utils import utils, plot_utils


# steps/compute.py
def compute_pca(n_samples:int=500, n_params:int=5, n_pca_components:int=10, output_path:str="output"):
    print("[ComputePCA] Running PCA (placeholder)")

    
    input_file = pd.read_csv(f"{output_path}/input_{n_samples}_{n_params}params.csv")
    output_file = pd.read_csv(f"{output_path}/output_{n_samples}_{n_params}params/resampled_all_pressure_traces_rv.csv")

    # Create direcoty for results
    if not os.path.exists(f"{output_path}/output_{n_samples}_{n_params}params/PCA"):
        os.makedirs(f"{output_path}/output_{n_samples}_{n_params}params/PCA")


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

    X_pca.to_csv(f'{output_path}/output_{n_samples}_{n_params}params/PCA/PCA.csv', index=False)

    # Concatenate the PCA components with the original data     
    df_pca = pd.concat([df, X_pca], axis=1)
    df_pca.to_csv(f'{output_path}/output_{n_samples}_{n_params}params/resampled_all_pressure_traces_rv_with_pca.csv', index=False)

    X_pca.hist(bins=30, figsize=(15, 13), layout=(5, 2), alpha=0.7, color='orange')
    plt.suptitle(f'Histograms of the First 1{n_pca_components} Principal Components')
    plt.savefig(f'{output_path}/output_{n_samples}_{n_params}params/figures/histograms_pca.png')

    output_parameters = os.path.join(output_path, f'output_{n_samples}_{n_params}params')

    # Plot the explained variance ratio
    plot_utils.plot_pca_explained_variance(pca, output_path=output_parameters)

    # Plot the PCA transformed data
    plot_utils.plot_pca_transformed(pca, X_scaled, output_path=output_parameters)

