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
from fast_ml.model_development import train_valid_test_split
from sklearn.model_selection import GridSearchCV
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
        # X_train = pd.read_parquet("C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\\dataset\\feature_engineered_data\\X_train_df.parquet")
        # X_test = pd.read_parquet("C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\\dataset\\feature_engineered_data\\X_test_df.parquet")
        # y_train = pd.read_parquet("C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\\dataset\\feature_engineered_data\\y_train.parquet")
        # y_test = pd.read_parquet("C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\\dataset\\feature_engineered_data\\y_test.parquet")
        dataset = pd.read_parquet("C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\\dataset\\cleaned_data.parquet")
        X_train, y_train, X_val, y_val, X_test, y_test = train_valid_test_split(df=dataset,
                                                                                target='price',
                                                                                train_size=0.6,
                                                                                valid_size=0.2,
                                                                                test_size=0.2,
                                                                                random_state=42)
        # for training
        # need to drop the price column which is the dependent or target feature
        # reset index 
        for data in [X_train, y_train, X_val, y_val, X_test, y_test]:
            data.reset_index(drop=True, inplace=True)

        # change y values to be series
        # y_train_series = pd.Series(y_train)
        # y_test_series = pd.Series(y_test)
        return X_train,X_test,X_val,y_train,y_test,y_val
    
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
        X_train,X_test,X_val,y_train,y_test,y_val = ingest_data.get_data()
        return X_train,X_test,X_val,y_train,y_test,y_val
    except Exception as e:
        logging.error(e)
        raise e
    return None
    
@asset
def data_validation_method(ingest_data):
    print("data validtion step")
    writer = WhyLabsWriter()
    X_train,X_test,X_val,y_train,y_test,y_val = ingest_data
    
    # split the train dataset into batches as well as batch dataset
    X_train_batch_1 = X_train.iloc[:1000, :]
    X_train_batch_2 = X_train.iloc[1000:2000, :]
    X_train_batch_3 = X_train.iloc[2000:3000, :]
    X_train_batch_4 = X_train.iloc[3000:4000, :]
    X_train_batch_5 = X_train.iloc[4000:, :]
    # split the tst data in batches
    X_test_batch_1 = X_test.iloc[:250, :]
    X_test_batch_2 = X_test.iloc[250:450, :]
    X_test_batch_3 = X_test.iloc[450:650, :]
    X_test_batch_4 = X_test.iloc[650:850, :]
    X_test_batch_5 = X_test.iloc[850:, :]
    # split for evaluation in to batches
    X_val_batch_1 = X_val.iloc[:250, :]
    X_val_batch_2 = X_val.iloc[250:450, :]
    X_val_batch_3 = X_val.iloc[450:650, :]
    X_val_batch_4 = X_val.iloc[650:850, :]
    X_val_batch_5 = X_val.iloc[850:, :]

    # batches for y
    y_train_batch_1 = y_train.iloc[:1000]
    y_train_batch_2 = y_train.iloc[1000:2000]
    y_train_batch_3 = y_train.iloc[2000:3000]
    y_train_batch_4 = y_train.iloc[3000:4000]
    y_train_batch_5 = y_train.iloc[4000:]
    # batches for y test
    y_test_batch_1 = y_test.iloc[:250]
    y_test_batch_2 = y_test.iloc[250:450]
    y_test_batch_3 = y_test.iloc[450:650]
    y_test_batch_4 = y_test.iloc[650:850]
    y_test_batch_5 = y_test.iloc[850:]
 
    # batches for y val
    y_val_batch_1 = y_val.iloc[:350]
    y_val_batch_2 = y_val.iloc[250:450]
    y_val_batch_3 = y_val.iloc[450:650]
    y_val_batch_4 = y_val.iloc[650:850]
    y_val_batch_5 = y_val.iloc[850:]


    X_train_list_batches = [X_train_batch_1,X_train_batch_2,X_train_batch_3,X_train_batch_4,X_train_batch_5]
    y_train_list_batches = [y_train_batch_1, y_train_batch_2, y_train_batch_3, y_train_batch_4, y_train_batch_5]

    X_test_list_batches = [X_test_batch_1,X_test_batch_2,X_test_batch_3,X_test_batch_4,X_test_batch_5]
    y_test_list_batches = [y_test_batch_1,y_test_batch_2, y_test_batch_3, y_test_batch_4, y_test_batch_5]

    X_val_list_batches = [X_val_batch_1, X_val_batch_2, X_val_batch_3, X_val_batch_4, X_val_batch_5]
    y_val_list_bacthes = [y_val_batch_1, y_val_batch_2, y_val_batch_3, y_val_batch_4, y_val_batch_5]
    # create whylog profiles for the datasets
    for i, X_train_batch in enumerate(X_train_list_batches):
        # calc diff in dates
        dt = datetime.now() - timedelta(days = i)
        # create profile for each day
        X_train_profile = why.log(X_train_batch).profile()
        #set time stamp for each batch or day data
        X_train_profile.set_dataset_timestamp(dt)
        # write the profile to the whylabs platform
        writer.write(file=X_train_profile.view())

    for i, X_test_batch in enumerate(X_test_list_batches):
        # calc diff in dates
        dt = datetime.now() - timedelta(days = i)
        # create profile for each day
        X_test_profile = why.log(X_test_batch).profile()
        #set time stamp for each batch or day data
        X_test_profile.set_dataset_timestamp(dt)
        # write the profile to the whylabs platform
        writer.write(file=X_test_profile.view())

    for i, X_val_batch in enumerate(X_val_list_batches):
        # calc diff in dates
        dt = datetime.now() - timedelta(days = i)
        # create profile for each day
        X_val_profile = why.log(X_val_batch).profile()
        #set time stamp for each batch or day data
        X_val_profile.set_dataset_timestamp(dt)
        # write the profile to the whylabs platform
        writer.write(file=X_test_profile.view())

    
    return X_train_list_batches, y_train_list_batches, X_test_list_batches, y_test_list_batches, X_val_list_batches, y_val_list_bacthes
    


model_n_score = []
def get_metrics(X_train, y_train, model):
  y_pred = model.predict(X_train)
  acc_val = r2_score(y_train, y_pred)
  mae_val = mean_absolute_error(y_train,y_pred)
  return acc_val,mae_val


@asset
def train_model(data_validation_method) -> tuple:
    
    print('Model Training')
    # experiment = tracking_datails_init()
    counter = 0
    train_metrics_dict = {}
    test_metrics_dict = {}
    # X_train,X_test,y_test_series,y_train_series = ingest_data
    X_train_list_batches, y_train_list_batches, X_test_list_batches, y_test_list_batches, _, _ = data_validation_method

    scaler = StandardScaler()
    mlmodel = GradientBoostingRegressor(n_estimators=500)
    
    column_transformer = make_column_transformer((OneHotEncoder(sparse=False), [0]), remainder='passthrough')
    model_pipeline = make_pipeline(column_transformer, scaler, mlmodel)
    for i, X_train_batch in enumerate(X_train_list_batches):
    
    #     mlmodel = model_class()
       
        y_train_batch = y_train_list_batches[i]
        y_test_batch = y_test_list_batches[i]
        X_test_batch = X_test_list_batches[i]
        # with experiment.train():
        model_pipeline.fit(X_train_batch,y_train_batch)
        accuracy_value, mae_value = get_metrics(X_train_batch, y_train_batch, model_pipeline)
        train_metrics_dict[f'train-accuracy :{counter}'] = accuracy_value
        train_metrics_dict[f'train-MAE :{counter}'] = mae_value
        val_acc , val_mae = get_metrics(X_test_batch,y_test_batch, model_pipeline)
          # experiment.log_metric('train-accuracy', accuracy_value)
          # experiment.log_metric('train-MAE', mae_value)
        counter += 1
    
    # experiment.log_metrics(train_metrics_dict)
    counter = 0

    # for j, X_test_batch in enumerate(X_test_list_batches):
    #     y_test_batch = y_test_list_batches[j]
        # with experiment.test():
          #X_train_pred = model_pipeline.predict(X_train_batch)
        # X_test_batch_transformed = column_transformer.transform(X_test_batch)
        # y_pred = model_pipeline.predict(X_test_batch)
        # acc_val = r2_score(y_test_batch, y_pred)
        # mae_val = mean_absolute_error(y_test_batch,y_pred)
        # test_metrics_dict[f'test-MAE :{counter}'] = mae_val
        #   # experiment.log_metric('test-accuracy', accuracy_value)
        #   # experiment.log_metric('test-MAE', mae_value)
        #   model_n_score.append([{'model_name':mlmodel,
        #                         'r2_score':accuracy_value}])
        # counter +=1

    # experiment.log_metrics(test_metrics_dict)
   
    
   
    # experiment.end()
    model_name = "house price pred"
    model_stage = "Development"
    model_version = '1.0.0'
    return model_name, model_stage, model_version, mlmodel


@asset
def model_tuning(train_model,data_validation_method):
    print("model validation")
    experiment = tracking_datails_init()
    
    X_train_list_batches, y_train_list_batches, X_test_list_batches, y_test_list_batches, _, _ = data_validation_method
    param_grid = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 500],
    'max_depth': [3, 5],
    'colsample_bytree': [0.5, 0.9],
    'gamma': [0, 0.1, 0.5],
    'reg_alpha': [0, 1, 10],
    'reg_lambda': [0, 1, 10],
}
    
    mlmodel = GradientBoostingRegressor(n_estimators=500)
    grid_search = GridSearchCV(mlmodel, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, 
                           scoring="neg_root_mean_squared_error", )

    scaler = StandardScaler()
    column_transformer = make_column_transformer((OneHotEncoder(sparse=False), [0]), remainder='passthrough')
    model_pipeline = make_pipeline(column_transformer, scaler, grid_search)

    for i, X_train_batch in enumerate(X_train_list_batches):
        y_train_batch = y_train_list_batches[i]
        with experiment.train():
            model_pipeline.fit(X_train_batch,y_train_batch)
            accuracy_value, mae_value = get_metrics(X_train_batch, y_train_batch, model_pipeline)
            metrics_dict[f'train-accuracy :{counter}'] = accuracy_value
            metrics_dict[f'train-MAE :{counter}'] = mae_value
            counter += 1
    
    counter = 0

    print("Best estimator: ", grid_search.best_estimator_)
    print("Best score: ", grid_search.best_score_)
    print("Best hyperparameters: ", grid_search.best_params_)

    return grid_search.best_estimator_, scaler, column_transformer

@asset
def model_validation(model_tuning, data_validation_method):
    print("model validation")
    experiment = tracking_datails_init()
    chosen_mlmodel,scaler, column_transformer = model_tuning
    _, _, _, _, X_val_list_batches, y_val_list_bacthes = data_validation_method

    model_pipeline = make_pipeline(column_transformer, scaler, chosen_mlmodel)
    counter = 0

    for j, X_val_batch in enumerate(X_val_list_batches):
        y_val_batch = y_val_list_bacthes[j]
        with experiment.test():
          accuracy_value, mae_value = get_metrics(X_val_batch, y_val_batch, model_pipeline)
          metrics_dict[f'test-accuracy :{counter}'] = accuracy_value
          metrics_dict[f'test-MAE :{counter}'] = mae_value
        counter +=1

    experiment.log_metrics(metrics_dict)

    experiment.end()

    return chosen_mlmodel


@asset
def model_registry(model_validation):
    print("model registry")
    model_name = "house price pred"
    model_stage = "Development"
    model_version = '1.0.0'
    chosen_mlmodel = model_validation
    experiment = tracking_datails_init()

    experiment.log_model(model_name, chosen_mlmodel)

    api = API()
    model = api.get_model(workspace='thobela', model_name=model_name)

    model.set_status(version=model_version, status=model_stage)
    model.add_tag(version=model_version, tag='new_model')



