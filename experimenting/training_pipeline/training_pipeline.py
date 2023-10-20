import pandas as pd
import numpy as np
import zenml
from zenml import pipeline, step


@pipeline
def training_pipeline(
    threshold,
    ingest_data,
    train_model,
    testing_model,
    # model_tuning,
    # model_validation,
    # model_registry,
    
    data_validation,
):
    
    # the steps for the DAG 
    # get data from get_data script
    #get_data()
    X_train,X_test,y_test,y_train= ingest_data()
    TRAINING_EXPERIMENT_NAME = train_model(X_train=X_train, y_train= y_train, X_val= X_test, y_val= y_test)
    r2, is_model_decent = testing_model(X_test, y_test, threshold)
    # model_tuning(X_train, y_train,X_test, y_test, is_model_decent)
    # model_validation(X_val= X_test, y_val= y_test)
    # model_registry()
    # MODEL_NAME = model_registering(TUNING_EXPERIMENT_NAME)
    data_validation(reference_dataset = X_train, target_dataset = X_test)
