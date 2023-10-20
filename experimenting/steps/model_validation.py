import comet_ml
from comet_ml import experiment
from comet_ml import API

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

import sklearn.base
from zenml.steps import step, Output, step_output

API_key = 'biu1KFYstI65GB8ztbszw9CdN'
proj_name = 'house-price-pred'
workspace = 'thobela'
model_name = "house-price-pred"
api = API()

@step()
def model_validation(X_train:pd.DataFrame, y_train:pd.DataFrame):
    # Applying k-Fold Cross Validation
    model = api.get_model(workspace=workspace, model_name=model_name)
    accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

