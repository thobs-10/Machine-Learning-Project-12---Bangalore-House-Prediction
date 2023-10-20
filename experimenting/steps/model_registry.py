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
def model_registry():
    model = api.get_model(workspace=workspace, model_name=model_name)

    model.set_status(version='1.1.0', status="Production")
    model.add_tag(version='1.1.0', tag='new_tag')
    
