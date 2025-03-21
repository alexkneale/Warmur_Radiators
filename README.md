# Models for Heat Output of Radiators

This project presents machine learning models designed to estimate the heat output of radiators, aiming to simplify the installation process by reducing the reliance on extensive radiator catalogues. We focus on three main radiator types—column radiators, panel radiators, and towel rails—with dedicated models developed for each due to the distinct underlying heat transfer characteristics. (Note: the towel rail model did not achieve sufficient accuracy because of limited data; for further details, please see the `towel_rail.ipynb` notebook.)

---

## Overview

- **Objective:**  
  Develop machine learning models that accurately predict the heat output (in Watts) of radiators, thereby aiding installers in the selection process.

- **Approach:**  
  - Separate models are built for column and panel radiators.
  - Towel rail models were explored but remain inconclusive due to insufficient data.
  - Models are initially developed and validated using Jupyter notebooks, then implemented in scikit-learn.
  - Final models are converted into ONNX format for deployment via C# inference.

---

## Project Components

### Jupyter Notebooks and Model Exploration

- **Core Notebooks:**  
  - `column_rad.ipynb`  
  - `panel_rad.ipynb`  
  - `towel_rail.ipynb`  

  In these notebooks, exploratory data analysis, feature selection, and model experiments are documented in detail. The notebooks also discuss various data processing strategies, machine learning models, and parameter choices. Users interested in the model development process should refer to these notebooks. If your focus is solely on executing the models, please see the sections below for instructions.

### Data Specifications

The project is based on a dataset provided in `Radiators.csv`, with the following characteristics:

- **Columns:**  
  `Type`, `Manufacturer`, `Range`, `Panel Radiator Type`, `Column Style`, `Material`, `Height`, `Width`, `Sections / Elements`, `Cols`, `Manu. Part Number`, `Heat Output Watts (dT50)`, `Heat Output Btu/hr (1)`, `n coefficient Strategy`, `n coefficient`.

- **Radiator Types in 'Type' Column:**  
  - Column: `"Column"`  
  - Panel: `"Panel"`  
  - Towel Rail: `"Towel Rail"`

- **Additional Details:**  
  - Column radiators feature two styles: `"Flat"` and `"Simple Tube"`.
  - Only the heat output in Watts is considered for modeling.

### Model Training and Deployment

- **Training Files:**  
  - **Column Radiators:**  
    - Notebook: `column_rad.ipynb`  
    - Pipeline script: `column_rads/column_model_pipeline.py`
  - **Panel Radiators:**  
    - Notebook: `panel_rad.ipynb`  
    - Pipeline script: `panel_rads/model_onnx.py`

  Both pipelines convert the trained scikit-learn models into ONNX format:
  - Column: `column_rads/column_model_pipeline.onnx`
  - Panel: `panel_rads/poly_model_pipeline.onnx`

- **Scaler Functions:**  
  The inverse scaling functions, which convert standardized outputs back to Watts, are saved as:
  - Column: `column_rads/inverse_target_scaler.onnx`
  - Panel: `panel_rads/inverse_target_scaler.onnx`

  Standardization was applied to enhance model stability.

- **Error Calibration:**  
  The model error, expressed as a percentage error based on test data, is recorded in JSON files:
  - Column: `column_rads/error_calibration.json`
  - Panel: `panel_rads/error_calibration.json`
  
  Error estimates are provided for the following heat output ranges (in Watts):  
  `(0 - 500)`, `(500 - 1000)`, `(1000 - 1500)`, `(1500 - 2000)`, `(2000, infinity)`.

### Validation and Testing

- **Model Verification:**  
  - Scripts:  
    - `column_rads/input_onnx.py`  
    - `panel_rads/panel_rad_input_onnx.py`

  These scripts execute the ONNX models and load the corresponding error data, allowing users to verify the correct functioning of the models before integrating them with the C# inference code.

### C# Inference

- **Deployment Folders:**  
  - `column_rad_app`  
  - `panel_rad_app`

  These folders contain the code for running the ONNX models via C# inference. Currently, input values (such as the number of columns and panels) must be manually set in the source code. Additionally, the C# application does not yet integrate the JSON error estimates. Future improvements should aim to incorporate these error estimates to provide more informative outputs.

---

## Final Remarks

This README outlines the architecture and rationale behind our radiator heat output prediction models. It is intended to serve as both a guide for further development and a reference for those integrating the models into production systems.

For detailed model exploration, analysis, and additional context, please refer to the corresponding Jupyter notebooks included in the project.
