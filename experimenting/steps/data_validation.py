import numpy as np
import pandas as pd

# from zenml.integrations.deepchecks.steps import (
#     DeepchecksDataDriftCheckStepParameters,
#     deepchecks_data_drift_check_step,
# )

LABEL_COL = "price"
	   	
# data_drift_detector = deepchecks_data_drift_check_step(
#     step_name="data_drift_detector",
#     params=DeepchecksDataDriftCheckStepParameters(
#         dataset_kwargs=dict(cat_features=[]),
#     ),
#)