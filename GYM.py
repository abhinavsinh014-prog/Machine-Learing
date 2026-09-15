import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

MUSCLE_PARAMS = {
    "Biceps":     {"Gmax": 6.0,  "k": 0.045, "mrv_time": 70,  "recovery_penalty": 0.020},
    "Triceps":    {"Gmax": 6.5,  "k": 0.040, "mrv_time": 75,  "recovery_penalty": 0.018},
    "Shoulders":  {"Gmax": 7.0,  "k": 0.030, "mrv_time": 90,  "recovery_penalty": 0.015},
    "Chest":      {"Gmax": 8.0,  "k": 0.026, "mrv_time": 100, "recovery_penalty": 0.014},
    "Back":       {"Gmax": 9.0,  "k": 0.020, "mrv_time": 130, "recovery_penalty": 0.010},
    "Quads":      {"Gmax": 10.0, "k": 0.017, "mrv_time": 150, "recovery_penalty": 0.009},
    "Hamstrings": {"Gmax": 8.5,  "k": 0.022, "mrv_time": 110, "recovery_penalty": 0.012},
    "Calves":     {"Gmax": 5.0,  "k": 0.035, "mrv_time": 80,  "recovery_penalty": 0.022},
}
 
MUSCLES = list(MUSCLE_PARAMS.keys())

print(MUSCLES)