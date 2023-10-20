import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd  # Add new imports to the top of `assets.py`
from dagster import (
    AssetKey,
    AssetIn,
    asset,
    get_dagster_logger,
    SourceAsset
)

# # get data from the preprocessing  pipeline
@asset(
    ins={"upstream": AssetIn("remove_outliers")}
)
def get_peprocessed_data(upstream:pd.DataFrame)-> pd.DataFrame:
    '''getting data from Preprocessing pipeline'''

    # features_df = pd.read_csv('preprocessed_data.parquet')
    # processed_df = remove_outliers()
    logger = get_dagster_logger()
    logger.info(f"getting data from Preprocessing pipeline")
    df = upstream.copy()
    return df

# feature transformation
# column tranforming the cat columns
@asset
def column_transform_categorical(get_peprocessed_data:pd.DataFrame)-> tuple:
    ''''column transformation for categoricals'''
    features_df = get_peprocessed_data.copy()
    #area type
    area_type_map=features_df['area_type'].value_counts().to_dict()
    features_df['area_type']=features_df['area_type'].map(area_type_map)
    # availability
    availability_map=features_df['availability'].value_counts().to_dict()
    features_df['availability']=features_df['availability'].map(availability_map)
    #location
    location_map=features_df['location'].value_counts().to_dict()
    features_df['location']=features_df['location'].map(location_map)
    # society
    society_map=features_df['society'].value_counts().to_dict()
    features_df['society']=features_df['society'].map(society_map)

    # seperate y value from the rest
    y = features_df.drop(columns=['area_type','availability','location','society','total_sqft','bath','balcony','bhk','price_per_sqft'])
    output_tuple = (features_df, y)
    return output_tuple

# feature importance
#### Remove The correlated

# find and remove correlated features

@asset
def computing_correlation_set(column_transform_categorical:tuple)-> set:
    '''computing column correlation'''
    threshold=0.8
    df = column_transform_categorical[0]
    dataset = df.copy()
    #drop the dependent value
    dataset.drop(columns=['price'], inplace=True)

    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


@asset
def drop_correlated_columns(computing_correlation_set:set, column_transform_categorical:tuple)-> pd.DataFrame:
    '''dropping correlated columns'''

    df = column_transform_categorical[0]
    val = computing_correlation_set.pop()
    df.drop(columns=[val], inplace=True)
    return df

@asset
def split_dataset(drop_correlated_columns:pd.DataFrame, column_transform_categorical:tuple)-> list:
    '''split the dataset for train and test'''
    # features present
    # ['area_type',	'availability',	'location',	'society',	'total_sqft',	'bath'	,'balcony',	'price_per_sqft']
    X = drop_correlated_columns.copy()
    y = column_transform_categorical[1]
    #split to train and test
    X_train, X_test ,y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
    split_data = [X_train, X_test ,y_train, y_test]
    return split_data

# scale/standardize/normilization
@asset
def scale_data(split_dataset:list)-> list:
    '''standardization of the dataset values'''
    X_train = split_dataset[0]
    X_test = split_dataset[1]
    scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    scaled_dataset_list = [X_train, X_test]
    return scaled_dataset_list

#save dataset
def save_engineered_data(scale_data:list, split_dataset:list)->None:

    X_train_scaled = scale_data[0]
    X_test_scaled = scale_data[1]

    y_train = split_dataset[2]
    y_test = split_dataset[3]

    X_train_scaled.to_parquet('datatse\\X_train_df.parquet')
    X_test_scaled.to_parquet('dataset\\X_test_df.parquet')
    y_train.to_csv('dataset\\y_train.parquet')
    y_test.to_csv('dataset\\y_test.parquet')



