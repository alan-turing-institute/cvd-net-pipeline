"""
Constants and configurations used across test modules.
"""

# Output keys for different test scenarios
OUTPUT_KEYS_FOR_TESTS_FOR_REAL_DATA_CALIBRATION = [
    "t_max_dpdt", 
    "a_epad",
    "epad", 
    "s_a_epad", 
    "s_epad",
    "min_dpdt", 
    "max_dpdt", 
    "A_p", 
    "P_max", 
    "esp", 
    "sys",
    "iT", 
    "PC1", 
    "PC2", 
    "PC3"
]

OUTPUT_KEYS_FOR_TESTS_FOR_SYNTHETIC_DATA_CALIBRATION = [
    "t_max_dpdt", 
    "a_epad", 
    "epad", 
    "s_a_epad", 
    "s_epad",
    "min_dpdt", 
    "max_dpdt", 
    "A_p", 
    "P_max", 
    "esp",
    "sys",
    "EF", 
    "Ees/Ea", 
    "iT", 
    "PC1", 
    "PC2", 
    "PC3"
]

# Common test parameters
DEFAULT_N_SAMPLES = 64
DEFAULT_N_PARAMS = 9
DEFAULT_EPSILON_OBS_SCALE = 0.05