import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from cvdnet_pipeline.utils import plot_utils


def compute_pca(n_samples:int=50, 
                n_params:int=9, 
                n_pca_components:int=10, 
                output_path:str="output", 
                data_type: str="synthetic"):
    
    output_file_name = 'waveform_resampled_all_pressure_traces_rv.csv' 

    if data_type == "synthetic":
           
        file_sufix = f'_{n_samples}_{n_params}_params'

        dir_name = f"{output_path}/output{file_sufix}"

    elif data_type == "real":

        dir_name = output_path

    output_file = pd.read_csv(f"{dir_name}/{output_file_name}")

    # Create directory for results
    if not os.path.exists(f"{dir_name}/pca"):
        os.makedirs(f"{dir_name}/pca")

    # Create directory for figures
    if not os.path.exists(f"{dir_name}/figures"):
        os.makedirs(f"{dir_name}/figures")

    ## Conduct PCA ##
    df = output_file.copy()

    # Copy the data and separate the target variable (only pressure traces)
    X = df.iloc[:,:100].copy() # traces only

    # Create an instance of the pipeline including StandardScaler and PCA
    pipeline = Pipeline(
        [ ('scl', StandardScaler()),
        ('pca', PCA(n_components=n_pca_components)),
        ('post',PowerTransformer())]
    )

    # Fit the pipeline to the data
    X_pca = pipeline.fit_transform(X)

    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names, index=df.index)

    X_pca.to_csv(f'{dir_name}/pca/PCA.csv', index=False)

    # Concatenate the PCA components with the original data     
    df_pca = pd.concat([df, X_pca], axis=1)

    # Create a new name for the output file which appends "_with_pca" to the original name
    output_file_name_pca = output_file_name.replace('.csv', '_with_pca.csv')

    df_pca.to_csv(f'{dir_name}/{output_file_name_pca}', index=False)

    # Plot the PCA histogram
    plot_utils.plot_pca_histogram(X_pca, output_path=dir_name, n_pca_components=n_pca_components)

    # Plot the explained variance ratio
    plot_utils.plot_pca_explained_variance(pipeline, output_path=dir_name)

    # Plot the PCA transformed data
    plot_utils.plot_pca_transformed(pipeline, X, output_path=dir_name)