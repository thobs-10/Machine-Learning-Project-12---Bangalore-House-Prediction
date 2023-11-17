import pandas as pd  # Add new imports to the top of `assets.py`
from dagster import (
    AssetKey,
    AssetIn,
    asset,
    get_dagster_logger,
    SourceAsset
)

# get preprocessing_data_profile
@asset(
    ins={"preprocessing_upstream": AssetIn("remove_outliers")}
)
def preprocessed_data_profile(preprocessing_upstream:pd.DataFrame):
    df = preprocessing_upstream.copy()
    df.to_parquet('C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\\dataset\\preprocessed_dataset\\preprocessed_data.parquet')
    

@asset(
    ins={"feature_engineering_upstream": AssetIn("scale_data")}
)
def feature_eng_data_profile(feature_engineering_upstream:list):
    X_train = feature_engineering_upstream[0]
    X_val = feature_engineering_upstream[1]
    X_test = feature_engineering_upstream[2]

    X_train.to_parquet('C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\dataset\\feature_engineered_data\\X_train_df.parquet')
    X_test.to_parquet('C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\dataset\\feature_engineered_data\\X_test_df.parquet')
    X_val.to_parquet('C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\dataset\\feature_engineered_data\\X_val_df.parquet')

@asset(
    ins={"split_data_upstream": AssetIn("split_dataset")}
)
def split_data_profile(split_data_upstream:list):
    y_train = split_data_upstream[3]
    y_val = split_data_upstream[4]
    y_test = split_data_upstream[5]

    y_train.to_csv('C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\dataset\\feature_engineered_data\\y_train.parquet')
    y_test.to_csv('C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\dataset\\feature_engineered_data\\y_test.parquet')
    y_val.to_csv('C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\dataset\\feature_engineered_data\\y_val.parquet')

