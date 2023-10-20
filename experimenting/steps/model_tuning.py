import comet_ml
from comet_ml import experiment
from comet_ml import API

import pandas as pd
import numpy as np

import sklearn.base
from zenml.steps import step, Output, step_output

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor

API_key = 'biu1KFYstI65GB8ztbszw9CdN'
proj_name = 'house-price-pred'
workspace = 'thobela'
model_name = "house-price-pred"
api = API()

@step()
def model_tuning(X_train:pd.DataFrame, y_train:pd.Series,X_test: pd.DataFrame, y_test:pd.Series, is_register: False):
    
    if is_register:
        column_transformer = make_column_transformer((OneHotEncoder(sparse=False), [0]), remainder='passthrough')
        scaler = StandardScaler()
        # get the model from the comet ml site
        model = api.get_model(workspace=workspace, model_name=model_name)
        # gradient boost reg
        grad_reg = GradientBoostingRegressor(n_estimators=500)
        grad_pipe = make_pipeline(column_transformer, scaler, grad_reg)
        # take the model and fine tune it using hyperparameter tuning
        # y_pred = grad_pipe.predict(X_test)
        # results
        rmse_train = np.sqrt(mean_squared_error(y_train, grad_pipe.predict(X_train)))
        rmse_test = np.sqrt(mean_squared_error(y_test, grad_pipe.predict(X_test)))
        print(f"Train RMSE: {rmse_train}, Test RMSE: {rmse_test}")

        # apply some tuning using grid searchCV
        from sklearn.model_selection import GridSearchCV
        param_grid = {
        'learning_rate': [0.001,0.01, 0.1],
        'n_estimators': [100, 500, 800],
        'max_depth': [3, 5],
        'colsample_bytree': [0.5, 0.9],
        'gamma': [0, 0.1, 0.5],
        'reg_alpha': [0, 1, 10],
        'reg_lambda': [0, 1, 10]}

        grid_search = GridSearchCV(grad_reg, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, 
                            scoring="neg_root_mean_squared_error", )

        grid_search.fit(X_train, y_train)
        #Fitting 5 folds for each of 432 candidates, totaling 2160 fits
        # get the best estimator
        best_model = grid_search.best_estimator_
        # get the params best for this estimator
        best_params = {}
        best_params['params'] = grid_search.best_params_
        # best score
        best_score = grid_search.best_score_

        y_pred = best_model.predict(X_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, best_model.predict(X_train)))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Train RMSE: {rmse_train}, Test RMSE: {rmse_test}")
    else:
        pass

    