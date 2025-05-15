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


# steps/build_emulator.py
def build_emulator(n_samples:int=500, n_params:int=5, output_path:str="output"):
    print("[BuildEmulator] Running PCA and training emulator (placeholder)")

    
    input_file = pd.read_csv(f"{output_path}/input_{n_samples}_{n_params}params.csv")
    output_file = pd.read_csv(f"{output_path}/output_{n_samples}_{n_params}params/resampled_all_pressure_traces_rv.csv")


    bool_exist = False
    if bool_exist:
        boolean_index = pd.read_csv(f"{output_path}/output_{n_samples}_{n_params}params/bool_indices_{n_samples}.csv")


build_emulator()