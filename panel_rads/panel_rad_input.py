from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

import json
import joblib


# load the fitted pipeline
loaded_model = joblib.load("panel_rads/poly_model_pipeline.pkl")

# load uncertainty dictionary
with open("panel_rads/error_calibration.json", "r") as f:
    error_dict = json.load(f)

# load target scaler, for inverse transformation
target_scaler = joblib.load("panel_rads/target_scaler.pkl")



# uncertainty calc function
def get_uncertainty(y_pred_value, error_dict):
    if 0 < y_pred_value < 500:
        return error_dict["0-500"]
    elif 500 <= y_pred_value < 1000:
        return error_dict["500-1000"]
    elif 1000 <= y_pred_value < 1500:
        return error_dict["1000-1500"]
    elif 1500 <= y_pred_value < 2000:
        return error_dict["1500-2000"]
    elif y_pred_value >= 2000:
        return error_dict["2000+"]
    else:
        return None  # if the value doesn't fall into any bin

Height = np.float64(input('Height = '))
Width = np.float64(input('Width = '))
Panels = int(input('Panels = '))
Fins = int(input('Fins = '))

X_observed = np.array([[Height,Width,Panels,Fins]])

# Use the loaded model for predictions
y_pred_scaled = loaded_model.predict(X_observed)
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_pred = y_pred.ravel()[0]


uncertainty_pct = get_uncertainty(y_pred,error_dict)*0.01
uncertainty = uncertainty_pct*y_pred

print(f'Warmur Prediction = {y_pred} +/- {uncertainty}')