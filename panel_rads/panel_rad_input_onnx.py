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
from skl2onnx import to_onnx
from onnx.reference import ReferenceEvaluator
import onnxruntime as ort
import numpy as np





# load uncertainty dictionary
with open("panel_rads/error_calibration.json", "r") as f:
    error_dict = json.load(f)


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

## 4 features for user to input
Height = np.float64(input('Height = '))
Width = np.float64(input('Width = '))
Panels = int(input('Panels = '))
Fins = int(input('Fins = '))



# Load the ONNX model
session = ort.InferenceSession("panel_rads/poly_model_pipeline.onnx")


# prepare input dictionary
input_dict = {
    'Height': np.array([[Height]], dtype=np.float32),
    'Width': np.array([[Width]], dtype=np.float32),
    'Panels': np.array([[Panels]], dtype=np.float32),
    'Fins': np.array([[Fins]], dtype=np.float32)
}
# model prediction, using input dictionary
onnx_pred = session.run(None, input_dict)[0]


# Load the inverse scaler ONNX model
session = ort.InferenceSession("panel_rads/inverse_target_scaler.onnx")

# Prepare your input data (make sure it's a 2D array, e.g., a column vector)
input_data = np.array(onnx_pred, dtype=np.float32)  

# Get the input name expected by the model
input_name = session.get_inputs()[0].name

# Run the inverse transformation
inverse_output = session.run(None, {input_name: input_data})[0]
# Extract scalar if needed (assuming single prediction)
inverse_value = inverse_output[0][0]


uncertainty_pct = get_uncertainty(inverse_value,error_dict)
uncertainty = uncertainty_pct*inverse_value/100

print(f'Warmur Prediction = {inverse_value} +/- {uncertainty}')