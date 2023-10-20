import numpy as np
import pandas as pd
from dagster import asset, get_dagster_logger # import the `dagster` library

# get data from the get data script
@asset
def get_data()-> pd.DataFrame:
    features_df = pd.read_csv('C:\\Users\\Cash Crusaders\\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\dataset\\archive (1)\\Bengaluru_House_Data.csv')

    # location
    features_df.dropna(subset=['location'],inplace=True)
    return features_df

# size
@asset
def impute_cat_nan(get_data) -> pd.DataFrame:
    cat_features_list = ['size','society']
    for feature in cat_features_list:
        most_frequent_category=get_data[feature].mode()[0]
        get_data[feature].fillna(most_frequent_category,inplace=True)
    
    return get_data

# for feature in ['size','society']:
#     impute_cat_nan(features_df,feature)

# change size to BHK
@asset
def bhk_transformation(impute_cat_nan)-> pd.DataFrame:
    df = impute_cat_nan.copy()
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
def apply_change_dtypes(bhk_transformation)-> pd.DataFrame:
    features_df = bhk_transformation.copy()
    features_df['total_sqft'] = features_df['total_sqft'].apply(convert_range)
    return features_df

# impute for numerical features
@asset
def impute_numeric_nan(apply_change_dtypes)-> pd.DataFrame:
    numeric_features_list = ['total_sqft','bath','balcony']
    df = apply_change_dtypes.copy()
    for numeric_col in numeric_features_list:
        column_median = df[numeric_col].median()
        df[numeric_col].fillna(column_median,inplace=True)
    
    return df

# for numeric_col in ['total_sqft','bath','balcony']:
#     impute_numeric_nan(features_df,numeric_col)
@asset
def feature_transformation(impute_numeric_nan)-> pd.DataFrame:
    df = impute_numeric_nan.copy()
    # create a new column called price per sqft which is made up of price and total sqft
    df['price_per_sqft'] = df['price'] * 100000/df['total_sqft']


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
    numeric_features_list = ['total_sqft','bath', 'balcony', 'bhk', 'price_per_sqft']
    df = feature_transformation.copy()
    for column_name in numeric_features_list:
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

#save dataset
@asset
def save_processed_data(remove_outliers)-> pd.DataFrame:
    df = remove_outliers.copy()
    df.to_parquet('preprocessed_data.parquet')
    return df

# def preprocessing_workflow() ->pd.DataFrame:
    # features_df = get_data()
    # #list of cat features
    
    # features_df = impute_cat_nan(features_df, cat_features_list)
    # features_df = bhk_transformation(features_df)
    # features_df = apply_change_dtypes(features_df)
    
    # features_df = impute_numeric_nan(features_df, numeric_features_list)
    # features_df = feature_transformation(features_df)
    # # outliers
    
    # features_df = remove_outliers(features_df,features_list)

    # return save_processed_data(features_df)


