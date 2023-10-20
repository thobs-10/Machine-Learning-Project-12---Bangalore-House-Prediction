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
