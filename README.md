# Models for Heat Output of Radiators

Here we attempt to produce machine learning models for radiator heat outputs. The rationale for this project is to attempt to help installers gauge a radiator's heat output, without having to consult endless radiator catalogues. 

There are three main types of radiators: column radiators, panel radiators and towel rails. A machine learning model was constructed for each type of radiator, as the underlying physics of heat output is quite different between radiator types. Therefore a 'universal' radiator model cannot be constructed. In our exploration, we found that we could not construct a sufficiently accurate towel rail model. This was mostly due to us having insufficient data. If more data is available in the future, please consult previous work done on towel rails in the jupyter notebook towel_rail.ipynb.

#### Exploration of Different ML Models in Jupyter Notebooks
The 'core' of the project revolves around the jupyter notebooks: column_rad.ipynb, panel_rad.ipynb and towel_rail.ipynb. In these notebooks, I have performed exploratory data analysis of our data and experimented with different models. I have included comments and discussions of different models and methodologies here. Therefore, anyone curious to understand the choices of data processing, ML models and parameters should consult these notebooks. That said, if you are only interested in running the model, without understanding the underlying decisions I have made, you do not have to consult these notebooks, and instead read the instructions in the next few sections. 

#### Note on data style

Throughout this project, we have assumed that the data we are working with to create ML models was of the following form (which it was at the time of the project).

- The dataset has the following columns: Type,	Manufacturer,	Range,	Panel Radiator Type,	Column Style,	Material,	Height,	Width	Sections / Elements	Cols,	Manu. Part Number,	Heat Output Watts (dT50),	Heat Output Btu/hr (1),	n coefficient Strategy,	n coefficient.
- Column radiators, panel radiators and towel radiators are denoted as the following under 'Type' column: 'Column', 'Panel', 'Towel Rail'
- 

#### Files that train and save the model

Sufficiently accurate models were constructed for column radiators and panel radiators. The most important code from column_rad.ipynb and panel_rad.ipynb was then transferred to panel_rads/model_onnx.py and column_rads/column_model_pipeline.py. In both of these two files, we use the all code from both jupyter notebooks which is sufficient to construct the machine learning models and gauge their accuracy. We then transform both of these scikit-learn models into ONNX models (panel_rads/poly_model_pipeline.onnx and column_rads/column_model_pipeline.onnx. Furthermore, we also save the functions that convert our standardized heat outputs (a number from 0 to 1) to unstandardized heat outputs in Watts to the files panel_rads/inverse_target_scaler.onnx and column_rads/inverse_target_scaler.onnx. A rationale for standardizing our heat outputs is that it allows for greater stability. Any reader should ignore the code in panel_rads/poly_model_pipeline.onnx that creates the file panel_rads/target_scaler.onnx. The section of the code and file were created in a previous version of the project, and are currently redundant. Finally, the error of the models, in the form of performance of the models on test data, was calibrated and saved to the json files column_rads/error_calibration.json and panel_rads/error_calibration.json. Here, the error was saved for the following ranges of true heat outputs: (0 - 500), (500 - 1000), (1000 - 1500), (1500 - 2000), ( 
2000, infinity). 

#### Files that double check everything is running fine




