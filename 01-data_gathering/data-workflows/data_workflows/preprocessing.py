import numpy as np
import requests
import pandas as pd  # Add new imports to the top of `assets.py`
from dagster import (
    AssetKey,
    DagsterInstance,
    MetadataValue,
    Output,
    asset,
    get_dagster_logger,
) # import the `dagster` library

# get data from the get data script
@asset
def get_data()-> tuple:
    '''fetch raw data'''
    df = pd.read_csv('C:\\Users\\Thobs\\Desktop\\Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\\dataset\\archive (1)\\Bengaluru_House_Data.csv')
    features_df = df.copy()
    y_df = df.drop(columns=['area_type','availability','location','size','society','total_sqft','bath','balcony'])
    # y_df.head()
    # location
    features_df.dropna(subset=['location'],inplace=True)
    output_tuple = (features_df, y_df)
    return output_tuple
# size
@asset
def impute_categorical_nan(get_data:tuple) -> pd.DataFrame:
    '''impute categorical null values'''
    df = get_data[0]
    cat_features_list = ['size','society']
    for feature in cat_features_list:
        most_frequent_category=df[feature].mode()[0]
        df[feature].fillna(most_frequent_category,inplace=True)
    
    return df

@asset
def bhk_transformation(impute_categorical_nan)-> pd.DataFrame:
    '''transform the bhk column'''
    df = impute_categorical_nan.copy()
    df['bhk'] = df['size'].str.split().str.get(0).astype(int)
    df.drop(columns=['size'], inplace=True)
    return df

# total sqft
# change the sq_ft datatype
def convert_range(x):
    #split the row value
    temp = x.split('-')
    # if the temp size is equal to 2
    if len(temp) == 2:
        return (float(temp[0]) + float(temp[1]))/2
    try:
        return float(x)
    except:
        return None

# apply the function
@asset
def apply_change_dtypes(impute_categorical_nan)-> pd.DataFrame:
    '''convert bhk datatypes'''
    #part 1
    df = impute_categorical_nan.copy()
    df['bhk'] = df['size'].str.split().str.get(0).astype(int)
    df.drop(columns=['size'], inplace=True)
    #part 2
    features_df = df.copy()
    features_df['total_sqft'] = features_df['total_sqft'].apply(convert_range)
    return features_df

# combine the dags
@asset
def combine_tranformation_changes(bhk_transformation:pd.DataFrame,apply_change_dtypes:pd.DataFrame):
    '''combine the changes to single dataframe'''
    df = apply_change_dtypes.copy()
    return df


# impute for numerical features

@asset
def impute_numeric_nan(combine_tranformation_changes)-> pd.DataFrame:
    '''impute numerical null values'''
    numeric_features_list = ['total_sqft','bath','balcony']
    df = combine_tranformation_changes.copy()
    for numeric_col in numeric_features_list:
        column_median = df[numeric_col].median()
        df[numeric_col].fillna(column_median,inplace=True)
    
    return df

# for numeric_col in ['total_sqft','bath','balcony']:
#     impute_numeric_nan(features_df,numeric_col)
@asset
def feature_transformation(impute_numeric_nan:pd.DataFrame, get_data:tuple)-> pd.DataFrame:
    '''transforming features'''
    df = impute_numeric_nan.copy()
    y_df = get_data[1]
    # create a new column called price per sqft which is made up of price and total sqft
    df['price_per_sqft'] = y_df['price'] * 100000/df['total_sqft']


    # usinng strip to remove the white spaces  before creating the location column
    df['location'] = df['location'].apply(lambda x : x.strip())
    location_count = df['location'].value_counts()

    # getting locations that are less than ten
    locattion_less_than_10 = location_count[location_count<10]

    df['location'] = df['location'].apply(lambda x : 'other' if x in locattion_less_than_10 else x)

    # we want the number, how many bhk for this specific total sqft
    df = df[((df['total_sqft']/df['bhk']) >= 300)]

    return df

# removee the outliers
@asset
def remove_outliers(feature_transformation)-> pd.DataFrame:
    '''removing outliers from numeric column'''
    features_list = ['total_sqft','bath', 'balcony', 'bhk', 'price_per_sqft']
    df = feature_transformation.copy()
    for column_name in features_list:
        # calculate the Quantiles(Q1 and Q3)
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        # calclulate the Inter_quatile_range IQR
        IQR = Q3 - Q1
        # calculate the lower limit and upper  limit (LL & UL)
        LL = Q1 - 1.5 * IQR
        UL = Q3 + 1.5 * IQR
        # now filter the column to remove the outliers
        # replace all the values that are less or equal to the LL in the hours per weeek column with the LL
        df.loc[df[column_name] <= LL, column_name] = LL
        # do the same for values greater than the UL
        df.loc[df[column_name] >= UL, column_name] = UL
    return df

def preprocessed_data_profile(remove_outliers)->None:
    df = remove_outliers.copy()
    df.to_parquet('preprocessed_data.parquet')
    return df

#save dataset

def save_processed_data(remove_outliers)-> pd.DataFrame:
    remove_outliers.to_parquet('preprocessed_data.parquet')
    return remove_outliers


def preprocessing_workflow() ->pd.DataFrame:
    # features_df = get_data()
    # #list of cat features
    # cat_features_list = ['size','society']
    # features_df = impute_cat_nan(features_df, cat_features_list)
    # features_df = bhk_transformation(features_df)
    # features_df = apply_change_dtypes(features_df)
    # numeric_features_list = ['total_sqft','bath','balcony']
    # features_df = impute_numeric_nan(features_df, numeric_features_list)
    # features_df = feature_transformation(features_df)
    # # outliers
    # features_list = ['total_sqft','bath', 'balcony', 'bhk', 'price_per_sqft']
    # features_df = remove_outliers(features_df,features_list)

    return save_processed_data()


