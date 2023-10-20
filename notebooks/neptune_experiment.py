import pandas as pd
import numpy as np

import neptune
import neptune.integrations.sklearn as npt_utils


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

def main():
    # get data 
    df = pd.read_parquet('cleaned_data.parquet')

    #seperate target from independent features
    X = df.drop(columns=['price'])
    y = df['price']

    # split train and test dataset
    X_train, X_test ,y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

    # declare some preprocessing classees
    column_transformer = make_column_transformer((OneHotEncoder(sparse=False), [0]), remainder='passthrough')
    scaler = StandardScaler()
    lin_reg = LinearRegression(normalize=True)

    # initialize neptune

    neptune_run = neptune.init_run(
            project="thobela.sixpence/house-price-pred",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZjI2OWZlZi1mZTNkLTQxMDItOGNlYy1iMjdlYzAwNGI1ZWUifQ==")
    # your credentials
    neptune_model = neptune.init_model(
            name="Prediction model",
            key="MOD", 
            project="thobela.sixpence/house-price-pred", 
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZjI2OWZlZi1mZTNkLTQxMDItOGNlYy1iMjdlYzAwNGI1ZWUifQ==")

    model_version = neptune.init_model_version(
            model="HPP-MOD",
            project="thobela.sixpence/house-price-pred",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZjI2OWZlZi1mZTNkLTQxMDItOGNlYy1iMjdlYzAwNGI1ZWUifQ==")


    # start logging training metrics
    # neptune_run["lin_reg_summary"] = npt_utils.create_regressor_summary(
    # lin_reg, X_train, X_test, y_train, y_test)

    # do tthe training 
    lin_pipe = make_pipeline(column_transformer, scaler, lin_reg)

    lin_pipe.fit(X_train,y_train)

    y_pred = lin_pipe.predict(X_test)

    print('R2 score',r2_score(y_test,y_pred))
    neptune_run['test/r2_score']= r2_score(y_test,y_pred)
    print('MAE',mean_absolute_error(y_test,y_pred))
    neptune_run['test/mean_absolute_error'] = mean_absolute_error(y_test,y_pred)
    neptune_run['model/weights'].upload('lin_reg.pkl')
    # stop the tracking
    neptune_run.stop()

if __name__ == "__main__":
    main()