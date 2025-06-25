import pandas as pd
from AnalysisGiessen import analyseGiessen

def analyse_giessen(file_path: str, gaussian_sigmas : list[float]):

    
    file = pd.read_csv(f"{file_path}/pressure_traces_rv/all_pressure_traces.csv")
    
    # unpack sigmas
    sigma_filter_pressure, sigma_filter_dpdt, sigma_filter_d2pdt2 = gaussian_sigmas

    # Here takes all Pressure traces from step 1

    all_pressure_traces = pd.DataFrame()
    for ind in range(len(file)):
        if ind % 1000 == 0:
            print(f"Processing {ind}th trace")
        dt = file.loc[ind, 'dt']
        f = file.iloc[[ind], :100].T
        f_repeated = pd.concat([f] * 5, axis=0, ignore_index=True)
        f_repeated.columns = ["Pressure"]
        f_repeated["cPressure"] = f_repeated['Pressure']

        ag = analyseGiessen(df=f_repeated, t_resolution=dt)
        ag.sigma_filter_pressure = sigma_filter_pressure
        ag.sigma_filter_dpdt     = sigma_filter_dpdt
        ag.sigma_filter_d2pdt2   = sigma_filter_d2pdt2

        ag.compute_derivatives()
        ag.compute_points_of_interest(height=10, use_filter=False) # , export_true_derivates=True, export_true_p=True, distance=90 (we should consder adding these options)
        beats = pd.DataFrame(ag.resample_heart_beat())
        sumstats = ag.points_df

        resampled_df = pd.concat([beats, sumstats.iloc[:-1, :]], axis=1)
        # overwrite EF with model version
        resampled_df['MC_EF'] = file.loc[ind, 'EF']
        
        all_pressure_traces = pd.concat([all_pressure_traces, resampled_df.iloc[[2]]], axis=0)

    all_pressure_traces.reset_index(drop=True, inplace=True)
    all_pressure_traces.to_csv(f"{file_path}/waveform_resampled_all_pressure_traces_rv.csv", index=False)
