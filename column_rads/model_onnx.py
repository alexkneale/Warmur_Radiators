### Libaries

from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer

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

### load dataset, and clean it

# load dataset
df = pd.read_csv('Radiators.csv')

# drop irrelevant features
features_to_drop = ['Range','Manufacturer','Material', 'Panel Radiator Type','Manu. Part Number','Heat Output Btu/hr (1)','n coefficient Strategy', 'n coefficient', 'Sections / Elements']
df = df.drop(features_to_drop, axis = 1)

# select only column rads
df = df.loc[df.Type == 'Column']
df = df.drop('Type', axis =1)

# rename some features, for ease
df = df.rename(columns = { 'Heat Output Watts (dT50)' : 'Heat', 'Column Style' : 'Column_Style' })# select rads that fit barry's descriptions 
df = df.loc[df.Height <= 750]
df = df.loc[df.Height >= 300]
df = df.loc[df.Width <= 1600]
df = df.loc[df.Width >= 400]

# manually binarize categorical feature Column_Style
df['Column_Style'] = df['Column_Style'].map({'Flat': 0, 'Simple Tube': 1})


### training and test datasets

# random seed for reproducibility
seed = 3851

X = df.drop(['Heat'], axis = 1)
features = list(X.columns)

y = df['Heat'].copy()
y = y.values

# training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

print(features)
# standardizing target variable 
y_train = y_train.reshape(-1,1)
target_scaler  = MinMaxScaler()
y_train_scaled = target_scaler.fit_transform(y_train)


### saving inverse scaler transformation

data_min = target_scaler.data_min_  # shape: (1,) for a single feature
data_max = target_scaler.data_max_
scale = data_max - data_min

# create input and output tensor info (here, working on a single feature)
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 1])
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 1])

# create constant tensors for scale and data_min
scale_tensor = helper.make_tensor("scale", TensorProto.FLOAT, [1], scale.astype(np.float32))
min_tensor = helper.make_tensor("min", TensorProto.FLOAT, [1], data_min.astype(np.float32))

# create a multiplication node: x * scale
mul_node = helper.make_node(
    'Mul',
    inputs=['input', 'scale'],
    outputs=['mul_output']
)

# create an addition node: (x * scale) + data_min
add_node = helper.make_node(
    'Add',
    inputs=['mul_output', 'min'],
    outputs=['output']
)

# build the graph
graph = helper.make_graph(
    [mul_node, add_node],
    "InverseMinMaxScaler",
    [input_tensor],
    [output_tensor],
    initializer=[scale_tensor, min_tensor]
)

opset_id = helper.make_operatorsetid("", 21)

# create the model
model = helper.make_model(graph, producer_name='inverse_minmax_scaler', opset_imports=[opset_id])
onnx.checker.check_model(model)

# save the ONNX model to file
onnx.save(model, "column_rads/inverse_target_scaler.onnx")

### model

# preprocessor
preprocessor = ColumnTransformer([
    ("cat_pre", OneHotEncoder(drop='if_binary'), ["Column_Style"]),
    ("count_pre", MinMaxScaler(), ["Height", "Width", "Cols"]),
])

# polynomial 3 model
pf = PolynomialFeatures(degree= 3)

# model pipeline
poly_interact_pipe_final = Pipeline([
        ("pre_processing", preprocessor),
        ('poly', pf),
        ('model', LinearRegression(fit_intercept=True))
])


poly_interact_fit = poly_interact_pipe_final.fit(X_train, y_train_scaled.ravel())

# assuming the model expects 4 input features
initial_type = [
    ('Column_Style', FloatTensorType([None, 1])),
    ('Height', FloatTensorType([None, 1])),
    ('Width', FloatTensorType([None, 1])),
    ('Cols', FloatTensorType([None, 1]))
]# convert loaded model
onnx_model = convert_sklearn(poly_interact_fit, initial_types=initial_type)
#saving onnx model
with open("column_rads/column_model_pipeline.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

### uncertainty dictionary

# estimating outputs from test inputs
y_pred_scaled = poly_interact_fit.predict(X_test)
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_pred = y_pred.ravel()

# percentage error, to use for uncertainty propogation
pct_diff = np.abs(y_test - y_pred) / y_test * 100

# define the bins as tuples (lower_bound, upper_bound) and corresponding labels.
bins = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, np.inf)]
bin_labels = ["0-500", "500-1000", "1000-1500", "1500-2000", "2000+"]

# uncertainty dictionary
error_dict = {}

for (lower, upper), label in zip(bins, bin_labels):
    if np.isinf(upper):
        mask = (y_test >= lower)
    else:
        if lower == 0:
            mask = (y_test > lower) & (y_test < upper)
        else:
            mask = (y_test >= lower) & (y_test < upper)
    # compute mean percentage error for the bin
    avg_error = np.mean(pct_diff[mask])
    error_dict[label] = avg_error

# save the error calibration to a JSON file for later use
with open("column_rads/error_calibration.json", "w") as f:
    json.dump(error_dict, f, indent=4)