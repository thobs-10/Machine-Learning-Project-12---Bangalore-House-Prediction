import comet_ml
from comet_ml import experiment
from comet_ml import API


import numpy as np
import pandas as pd
import pickle

import neptune

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor



# API_key = 'biu1KFYstI65GB8ztbszw9CdN'
# proj_name = 'house-price-prediction'

# experiment = comet_ml.Experiment(
#     api_key=API_key,
#     project_name=proj_name,
#      workspace="thobela",
# )


metrics_dict = {}
counter = 0
model_n_score = []
def get_metrics(X_train, y_train, model):
  y_pred = model.predict(X_train)
  acc_val = r2_score(y_train, y_pred)
  mae_val = mean_absolute_error(y_train,y_pred)
  return acc_val,mae_val



def train_model(X_train:pd.DataFrame, y_train:pd.Series, X_test:pd.DataFrame, y_test:pd.Series):
    
    print('Model Training')
    # scaler = StandardScaler()
    # column_transformer = make_column_transformer((OneHotEncoder(sparse=False), [0]), remainder='passthrough')
    # for model_class in (RandomForestRegressor,GradientBoostingRegressor
    #                     ,AdaBoostRegressor,
    #                     ExtraTreesRegressor,SVR,DecisionTreeRegressor,KNeighborsRegressor,
    #                     LinearRegression,Ridge,Lasso
    #                     ):
    #     #initialize the neptune run to log metadata and parameters
    #     # neptune_run = neptune.init_run()
    #     # run.start()
    #     counter +=1
    #     mlmodel = model_class()
    #     model_pipeline = make_pipeline(column_transformer, scaler, mlmodel)

    #     with experiment.train():
    #       model_pipeline.fit(X_train,y_train)
    #       accuracy_value, mae_value = get_metrics(X_train, y_train, model_pipeline)
    #       metrics_dict[f'train-accuracy :{counter}'] = accuracy_value
    #       metrics_dict[f'train-MAE :{counter}'] = mae_value
    #       # experiment.log_metric('train-accuracy', accuracy_value)
    #       # experiment.log_metric('train-MAE', mae_value)

    #     with experiment.test():
    #       accuracy_value, mae_value = get_metrics(X_test, y_test, model_pipeline)
    #       metrics_dict[f'test-accuracy :{counter}'] = accuracy_value
    #       metrics_dict[f'test-MAE :{counter}'] = mae_value
    #       # experiment.log_metric('test-accuracy', accuracy_value)
    #       # experiment.log_metric('test-MAE', mae_value)
    #       model_n_score.append([{'model_name':mlmodel,
    #                             'r2_score':accuracy_value}])

    # experiment.log_metrics(metrics_dict)
    # max_value = -1
    # total_max = -1
    # for i in range(len(model_n_score)):
    #   max_value = -10000000
    #   cur_model_n_score_list = model_n_score[i]
    #   cur_dict = cur_model_n_score_list[0]
    #   if cur_dict['r2_score']*100 > max_value:
    #     max_value = cur_dict['r2_score']
    #     if max_value *100 > total_max *100:
    #       total_max = max_value
    #       better_model = cur_dict['model_name']
    #     else:
    #       continue
          
    #   else:
    #     continue


    # pickle.dump(better_model,open("house-model.pkl",'wb'))

    # model = pickle.load(open("house-model.pkl",'rb'))

    # # log the better moddel  on model resgistry
    # experiment.log_model("house price pred",'house-model.pkl')

    # # register the logged model into model registry
    # experiment.register_model("house price pred")

    # # track the version of the model
    # api = API(api_key=API_key)
    # model_name = "house price pred"
    # model = api.get_model(workspace="thobela", model_name='house-price-pred')

    # model.set_status(version='1.0.0', status="Development")
    # model.add_tag(version='1.0.0', tag='first version of the model')

    # experiment.end()
    

