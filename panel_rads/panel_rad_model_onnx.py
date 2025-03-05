### Libaries

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
from sklearn.preprocessing import FunctionTransformer

import json
import joblib
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
import onnx

from onnx import helper, TensorProto


loaded_model = joblib.load("panel_rads/poly_model_pipeline.pkl")
# assuming the model expects 4 input features
initial_type = [('input', FloatTensorType([None, 4]))]
# convert loaded model
onnx_model = convert_sklearn(loaded_model, initial_types=initial_type)
#saving onnx model
with open("panel_rads/poly_model_pipeline.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# load fitted target scaler 
target_scaler = joblib.load("panel_rads/target_scaler.pkl")

# define the input type
initial_type = [('input', FloatTensorType([None, 1]))]

# convert the target scaler to ONNX format
onnx_target_scaler = convert_sklearn(target_scaler, initial_types=initial_type)

# save the converted ONNX model to a file
with open("panel_rads/target_scaler.onnx", "wb") as f:
    f.write(onnx_target_scaler.SerializeToString())


## inverse scaler transformation

# Assume target_scaler is already loaded from joblib
data_min = target_scaler.data_min_  # shape: (1,) for a single feature
data_max = target_scaler.data_max_
scale = data_max - data_min

# Create input and output tensor info (here, working on a single feature)
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 1])
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 1])

# Create constant tensors for scale and data_min
scale_tensor = helper.make_tensor("scale", TensorProto.FLOAT, [1], scale.astype(np.float32))
min_tensor = helper.make_tensor("min", TensorProto.FLOAT, [1], data_min.astype(np.float32))

# Create a multiplication node: x * scale
mul_node = helper.make_node(
    'Mul',
    inputs=['input', 'scale'],
    outputs=['mul_output']
)

# Create an addition node: (x * scale) + data_min
add_node = helper.make_node(
    'Add',
    inputs=['mul_output', 'min'],
    outputs=['output']
)

# Build the graph
graph = helper.make_graph(
    [mul_node, add_node],
    "InverseMinMaxScaler",
    [input_tensor],
    [output_tensor],
    initializer=[scale_tensor, min_tensor]
)

opset_id = helper.make_operatorsetid("", 21)

# Create the model
model = helper.make_model(graph, producer_name='inverse_minmax_scaler', opset_imports=[opset_id])
onnx.checker.check_model(model)

# Save the ONNX model to file
onnx.save(model, "panel_rads/inverse_target_scaler.onnx")
