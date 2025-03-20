# Models for Heat Output of Radiators

Here we attempt to produce machine learning models for radiator heat outputs. The rationale for this project is to attempt to help installers gauge a radiator's heat output, without having to consult endless radiator catalogues. 

There are three main types of radiators: column radiators, panel radiators and towel rails. A machine learning model was constructed for each type of radiator, as the underlying physics of heat output is quite different between radiator types. Therefore a 'universal' radiator model cannot be constructed. In our exploration, we found that we could not construct a sufficiently accurate towel rail model. This was mostly due to us having insufficient data. If more data is available in the future, please consult previous work done on towel rails in the jupyter notebook towel_rail.ipynb.

#### Exploration of Different ML Models in Jupyter Notebooks
The 'core' of the project revolves around the jupyter notebooks: column_rad.ipynb, panel_rad.ipynb and towel_rail.ipynb. In these notebooks, I have performed exploratory data analysis of our data and experimented with different models. I have included comments and discussions of different models and methodologies here. Therefore, anyone curious to understand the choices of data processing, ML models and parameters should consult these notebooks. That said, if you are only interested in running the model, without understanding the underlying decisions I have made, you do not have to consult these notebooks, and instead read the instructions in the next few sections. 

#### Note on data style

Throughout this project, we have assumed that the data we are working with to create ML models was of the following form (which it was at the time of the project).
- 

#### Files that train and save the model: 

Sufficiently accurate models were constructed for column radiators and panel radiators. The most important code from column_rad.ipynb and panel_rad.ipynb was then transferred to panel_rads/model_onnx.py and column_rads/model_onnx.py. 




