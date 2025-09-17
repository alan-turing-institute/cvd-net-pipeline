"""
Constants and configurations used across the pipeline.
"""

VALID_PIPELINE_STEPS = ["sim", # simulation
                  "ag", # analyse giessen
                  "pca", # principal component analysis
                  "emu", # building emulations
                  "cal", # calibration
                  "kf", # Kalman filter
                  "post_sim", # posterior simulation
                  "post_res"] # posterior resampling