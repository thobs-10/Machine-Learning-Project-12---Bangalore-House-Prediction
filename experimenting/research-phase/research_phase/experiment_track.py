import numpy as np
import pandas as pd  # Add new imports to the top of `assets.py`
from dagster import (
    AssetKey,
    AssetIn,
    asset,
    get_dagster_logger,
    SourceAsset
)
# get data from the dataset folder 
import logging
import whylogs as why
from whylogs.api.writer.whylabs import WhyLabsWriter
import os
from datetime import datetime
from datetime import timedelta

import comet_ml
from comet_ml import experiment
from comet_ml import API


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
# details for logging in whylabs
os.environ["WHYLABS_DEFAULT_ORG_ID"] = "org-M3RPnk" # ORG-ID is case sensitive
os.environ["WHYLABS_API_KEY"] = "zWmRt0Qm9i.gepzx9yBhQabyD6c0IqQSqzLXbQxeuJJrW7sfPBWr7vf0L6DA5f2q:org-M3RPnk"
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "model-5" # The selected project "lending_club_credit_model (model-0)" is "model-0"
# details for tracking

def tracking_datails_init():
    API_key = 'biu1KFYstI65GB8ztbszw9CdN'
    proj_name = 'house-price-prediction'
    experiment = comet_ml.Experiment(
        api_key=API_key,
        project_name=proj_name,
        workspace="thobela",
    )
    return experiment

class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self) -> None:
        """Initialize the data ingestion class."""
        pass

    def get_data(self):
        X_train = pd.read_parquet("C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\\dataset\\feature_engineered_data\\X_train_df.parquet")
        X_test = pd.read_parquet("C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\\dataset\\feature_engineered_data\\X_test_df.parquet")
        y_train = pd.read_parquet("C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\\dataset\\feature_engineered_data\\y_train.parquet")
        y_test = pd.read_parquet("C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\\dataset\\feature_engineered_data\\y_test.parquet")
        # for training
        # need to drop the price column which is the dependent or target feature
        X_train.drop(columns=['price'],inplace=True)
        X_test.drop(columns=['price'],inplace=True)
        # change y values to be series
        # y_train_series = pd.Series(y_train)
        # y_test_series = pd.Series(y_test)
        return X_train,X_test,y_test,y_train
    
@asset
def ingest_data()-> tuple:
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    print("getting data from data repo/feature store")
    try:
        ingest_data = IngestData()
        X_train,X_test,y_test_series,y_train_series = ingest_data.get_data()
        return X_train,X_test,y_test_series,y_train_series
    except Exception as e:
        logging.error(e)
        raise e
    return None
    
@asset
def data_validation_method(ingest_data):
    print("data validtion step")
    writer = WhyLabsWriter()
    X_train,X_test,y_test_series,y_train_series = ingest_data
    
    # split the train dataset into batches as well as batch dataset
    X_train_batch_1 = X_train.iloc[:2000, :]
    X_train_batch_2 = X_train.iloc[2000:4000, :]
    X_train_batch_3 = X_train.iloc[4000:6000, :]
    X_train_batch_4 = X_train.iloc[6000:8000, :]
    X_train_batch_5 = X_train.iloc[8000:, :]

    # batches for y
    y_train_batch_1 = y_train_series.iloc[:2000, :]
    y_train_batch_2 = y_train_series.iloc[2000:4000, :]
    y_train_batch_3 = y_train_series.iloc[4000:6000, :]
    y_train_batch_4 = y_train_series.iloc[6000:8000, :]
    y_train_batch_5 = y_train_series.iloc[8000:, :]
 
    X_train_list_batches = [X_train_batch_1,X_train_batch_2,X_train_batch_3,X_train_batch_4,X_train_batch_5]
    y_train_list_batches = [y_train_batch_1, y_train_batch_2, y_train_batch_3, y_train_batch_4, y_train_batch_5]
    # create whylog profiles for the datasets
    for i, X_train_batch in enumerate(X_train_list_batches):
        # calc diff in dates
        dt = datetime.now() - timedelta(days = i)
        # create profile for each day
        profile = why.log(X_train_batch).profile()
        #set time stamp for each batch or day data
        profile.set_dataset_timestamp(dt)
        # write the profile to the whylabs platform
        writer.write(file=profile.view())
    return X_train_list_batches, y_train_list_batches
    
metrics_dict = {}

model_n_score = []
def get_metrics(X_train, y_train, model):
  y_pred = model.predict(X_train)
  acc_val = r2_score(y_train, y_pred)
  mae_val = mean_absolute_error(y_train,y_pred)
  return acc_val,mae_val


@asset
def train_model(data_validation_method,ingest_data) -> tuple:
    
    print('Model Training')
    experiment = tracking_datails_init()
    counter = 0

    X_train,X_test,y_test_series,y_train_series = ingest_data
    X_baches_list, y_batches_list = data_validation_method

    scaler = StandardScaler()
    mlmodel = GradientBoostingRegressor(n_estimators=500)
    
    column_transformer = make_column_transformer((OneHotEncoder(sparse=False), [0]), remainder='passthrough')
    model_pipeline = make_pipeline(column_transformer, scaler, mlmodel)
    for i, X_train_batch in enumerate(X_baches_list):
    
    #     mlmodel = model_class()
       
        y_train_batch = y_batches_list[i]
        with experiment.train():
          model_pipeline.fit(X_train_batch,y_train_batch)
          accuracy_value, mae_value = get_metrics(X_train_batch, y_train_batch, model_pipeline)
          metrics_dict[f'train-accuracy :{counter}'] = accuracy_value
          metrics_dict[f'train-MAE :{counter}'] = mae_value
          # experiment.log_metric('train-accuracy', accuracy_value)
          # experiment.log_metric('train-MAE', mae_value)
          counter += 1

        # with experiment.test():
        #   accuracy_value, mae_value = get_metrics(X_test, y_test, model_pipeline)
        #   metrics_dict[f'test-accuracy :{counter}'] = accuracy_value
        #   metrics_dict[f'test-MAE :{counter}'] = mae_value
        #   # experiment.log_metric('test-accuracy', accuracy_value)
        #   # experiment.log_metric('test-MAE', mae_value)
        #   model_n_score.append([{'model_name':mlmodel,
        #                         'r2_score':accuracy_value}])

    experiment.log_metrics(metrics_dict)
   


    # pickle.dump(better_model,open("house-model.pkl",'wb'))

    # model = pickle.load(open("house-model.pkl",'rb'))

    # # log the better moddel  on model resgistry
    # experiment.log_model("house price pred",'house-model.pkl')

    # # register the logged model into model registry
    # experiment.register_model("house price pred")

    # # track the version of the model
    # api = API(api_key=API_key)
    model_name = "house price pred"
    # model = api.get_model(workspace="thobela", model_name='house-price-pred')

    # model.set_status(version='1.0.0', status="Development")
    # model.add_tag(version='1.0.0', tag='first version of the model')

    experiment.end()

    model_stage = "Development"
    model_version = '1.0.0'
    return model_name, model_stage, model_version


@asset
def model_testing(train_model, ingest_data):
    print("model Testing")

@asset
def model_tuning(model_testing):
    print("model validation")

@asset
def model_validation(ingest_data,model_tuning):
    print("model validation")

@asset
def model_registry(model_validation):
    print("model registry")



