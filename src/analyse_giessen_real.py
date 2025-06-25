import pandas as pd
from AnalysisGiessen import analyseGiessen

def analyse_giessen_real(file_path: str):

    
    file = pd.read_csv(f"{file_path}/pressure_traces_rv/all_pressure_traces.csv")

    
    file[["Pressure", "cPressure"]] = file[["Pressure [mmHg]", "Compensated Pressure [mmHg]"]]

    ag = analyseGiessen(df=file)

    ag.compute_derivatives()
    ag.compute_points_of_interest()
    beats = pd.DataFrame(ag.resample_heart_beat())
    sumstats = ag.points_df

    resampled_df = pd.concat([beats, sumstats.iloc[:-1, :]], axis=1)
    resampled_df.to_csv(f"{file_path}/waveform_resampled_all_pressure_traces_rv.csv", index=False)
