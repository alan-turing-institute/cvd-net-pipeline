import pandas as pd
from giessen_data_analysis import analyseGiessen

def analyse_giessen(file_path: str, data_type: str, gaussian_sigmas : list[float]):

    if data_type == 'synthetic':
        rv_file = pd.read_csv(f"{file_path}/pressure_traces_rv/all_pressure_traces.csv")
        ar_file = pd.read_csv(f"{file_path}/pressure_traces_pat/all_pressure_traces.csv")
        
        # Flag and remove any negative pressure traces
        negative_indices = rv_file[(rv_file < 0).any(axis=1)].index
        if not negative_indices.empty:
            print(f"Removing {len(negative_indices)} negative RV pressure traces at indices: {negative_indices.tolist()}")
            rv_file = rv_file.drop(index=negative_indices).reset_index(drop=True)
        
        # unpack sigmas
        sigma_filter_pressure, sigma_filter_dpdt, sigma_filter_d2pdt2 = gaussian_sigmas

        # Here takes all Pressure traces from step 1

        all_pressure_traces = pd.DataFrame()
        for ind in range(len(rv_file)):
            if ind % 1000 == 0:
                print(f"Processing {ind}th trace")
            dt = rv_file.loc[ind, 'dt']
            f = rv_file.iloc[[ind], :100].T
            f_repeated = pd.concat([f] * 5, axis=0, ignore_index=True)
            f_repeated.columns = ["Pressure"]
            f_repeated["cPressure"] = f_repeated['Pressure']

            ag = analyseGiessen(df=f_repeated, t_resolution=dt)
            ag.sigma_filter_pressure = sigma_filter_pressure
            ag.sigma_filter_dpdt     = sigma_filter_dpdt
            ag.sigma_filter_d2pdt2   = sigma_filter_d2pdt2

            ag.compute_derivatives()
            # Should change start_at_edp bask to False
            ag.compute_points_of_interest(height=10, use_filter=False, start_at_edp=False) # , export_true_derivates=True, export_true_p=True, distance=90 (we should consder adding these options)
            beats = pd.DataFrame(ag.resample_heart_beat())
            sumstats = ag.points_df

            resampled_df = pd.concat([beats, sumstats.iloc[:-1, :]], axis=1)
            # overwrite EF with model version
            resampled_df['MC_EF'] = rv_file.loc[ind, 'EF']
            # add MC version of edp
            resampled_df['MC_edp'] = rv_file.loc[ind, "0"]
            # add MC version of eivc
            f_pat = ar_file.iloc[[ind],:100].T
            min_ind = f_pat.idxmin().values[0]
            resampled_df['MC_eivc'] = rv_file.loc[ind, min_ind]
            # add MC version of dia
            resampled_df['MC_dia'] = rv_file.loc[ind].min()
            # add t_pulse
            resampled_df['t_pulse'] = (100 + resampled_df['dia_ind'] - resampled_df['edp_ind']) * dt
            
            all_pressure_traces = pd.concat([all_pressure_traces, resampled_df.iloc[[2]]], axis=0)

        all_pressure_traces.reset_index(drop=True, inplace=True)
        all_pressure_traces.to_csv(f"{file_path}/waveform_resampled_all_pressure_traces_rv.csv", index=False)

    elif data_type == 'real':
        
        rv_file = pd.read_csv(f"{file_path}/raw_data.csv")

         # Here takes all Pressure traces from step 1
        rv_file[["Pressure", "cPressure"]] = rv_file[["Pressure [mmHg]", "Compensated Pressure [mmHg]"]]

        ag = analyseGiessen(df=rv_file)

        ag.compute_derivatives()
        ag.compute_points_of_interest()
        beats = pd.DataFrame(ag.resample_heart_beat())
        sumstats = ag.points_df

        resampled_df = pd.concat([beats, sumstats.iloc[:-1, :]], axis=1)
        resampled_df.to_csv(f"{file_path}/waveform_resampled_all_pressure_traces_rv.csv", index=False)