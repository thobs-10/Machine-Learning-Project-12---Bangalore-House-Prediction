import numpy as np
import pandas  as pd

# get data from the dataset folder 
import pandas as pd
import numpy as np
import logging


class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self) -> None:
        """Initialize the data ingestion class."""
        pass

    def get_data(self):
        X_train = pd.read_parquet("C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\\dataset\\feature_engineered_data\\X_train_df.parquet")
        X_test = pd.read_parquet("C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\\dataset\\feature_engineered_data\\X_test_df.parquet")
        y_train = pd.read_parquet("C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\\dataset\\feature_engineered_data\\y_train.parquet")
        y_test = pd.read_parquet("C:\\Users\\Cash Crusaders\\Desktop\\My Portfolio\\Projects\\Data Science Projects\\Machine Learning Project 12 - Bangalore House Prediction\\dataset\\feature_engineered_data\\y_test.parquet")
        # for training
        # need to drop the price column which is the dependent or target feature
        X_train.drop(columns=['price'],inplace=True)
        X_test.drop(columns=['price'],inplace=True)
        # change y values to be series
        # y_train_series = pd.Series(y_train)
        # y_test_series = pd.Series(y_test)
        return X_train,X_test,y_test,y_train
    

def ingest_data()->tuple(
        X_train = pd.DataFrame,
        X_test = pd.DataFrame,
        y_test =pd.DataFrame,
        y_train = pd.DataFrame
):
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData()
        X_train,X_test,y_test_series,y_train_series = ingest_data.get_data()
        return X_train,X_test,y_test_series,y_train_series
    except Exception as e:
        logging.error(e)
        raise e
