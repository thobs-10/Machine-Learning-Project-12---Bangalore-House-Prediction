import comet_ml
from comet_ml import experiment
from comet_ml import API

import pandas as pd
import numpy as np

import sklearn.base


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import r2_score, mean_absolute_error


# API_key = 'biu1KFYstI65GB8ztbszw9CdN'
# proj_name = 'house-price-pred'
# workspace = 'thobela'
# proj_name = 'house-price-prediction'
# api = API()
# experiment = comet_ml.Experiment(
#     api_key=API_key,
#     project_name=proj_name,
#      workspace="thobela",
# )

def get_metrics(X_train, y_train, model):
  y_pred = model.predict(X_train)
  acc_val = r2_score(y_train, y_pred)
  mae_val = mean_absolute_error(y_train,y_pred)
  return acc_val,mae_val


def testing_model( X_test: pd.DataFrame, y_test:pd.DataFrame,threshold : float):

  print('Model Testing')
    # is_model_decent = False
    # # get the model from the comet ml site
    # model = api.get_model(workspace=workspace, model_name=model_name)
    # # get the scaler to scale the testing dataset
    # scaler = StandardScaler()
    # # create a pipeline to make the testing prediction easier
    # model_pipeline = make_pipeline(scaler, model)
    # # log the testing process and also save the results in a csv file
    # with experiment.test():    
    #     # predictions = model_pipeline.predict(X_test)
    #     r2, mae = get_metrics(X_test, y_test, model_pipeline)
    #     experiment.log_metric(r2)
    #     experiment.log_metric(mae)
        
    # if r2 >= threshold:
    #    is_model_decent = True
    
    # return r2, is_model_decent
