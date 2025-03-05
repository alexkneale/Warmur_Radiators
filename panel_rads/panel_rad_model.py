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

import json
import joblib


### load dataset, and clean it

# load dataset
df = pd.read_csv('Radiators.csv')

# drop irrelevant features
features_to_drop = ['Sections / Elements','Manufacturer','Range','Column Style', 'Material','Cols','Manu. Part Number','Heat Output Btu/hr (1)','n coefficient Strategy', 'n coefficient']
df = df.drop(features_to_drop, axis = 1)

# select only panel rads
df = df.loc[df.Type == 'Panel']
df = df.drop('Type', axis =1)

# rename some features, for ease
df = df.rename(columns = {'Heat Output Watts (dT50)' : 'Heat', 'Panel Radiator Type' : 'panel_radiator_type' })

# select rads that fit barry's descriptions 
df = df.loc[df.Height <= 750]
df = df.loc[df.Height >= 300]
df = df.loc[df.Width <= 1600]
df = df.loc[df.Width >= 400]

# convert feature panel_radiator_type to two features: fins and panels

df['panel_radiator_type'] = df['panel_radiator_type'].astype(int)
df['panels'] = df['panel_radiator_type'] // 10
df['fins'] = df['panel_radiator_type'] % 10
df = df.drop('panel_radiator_type', axis = 1)


### training and test datasets

# random seed, for reproducibility
seed = 4542

X = df.drop(['Heat'], axis = 1)
features = list(X.columns)
X = X.values

y = df['Heat'].copy()
y = y.values

# training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Convert input arrays to double precision
X_train = X_train.astype(np.float64)
y_train = y_train.astype(np.float64)
X_test = X_test.astype(np.float64)
y_test = y_test.astype(np.float64)

# standardizing target variable 
y_train = y_train.reshape(-1,1)
target_scaler  = MinMaxScaler()
y_train_scaled = target_scaler.fit_transform(y_train)


# save the target_scaler to a file in the current directory
joblib.dump(target_scaler, "panel_rads/target_scaler.pkl")



### model

# polynomial 3 model
pf = PolynomialFeatures(degree= 3)

# model pipeline
poly_interact_pipe_final = Pipeline([
        ("count_pre", MinMaxScaler()), # Applied to the count variables
        ('poly', pf),
        ('model', LinearRegression(fit_intercept=True))
])

# fitting model
poly_interact_fit = poly_interact_pipe_final.fit(X_train, y_train_scaled.ravel())
# saving fitted model for later use
joblib.dump(poly_interact_fit, "panel_rads/poly_model_pipeline.pkl")


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
with open("panel_rads/error_calibration.json", "w") as f:
    json.dump(error_dict, f, indent=4)


