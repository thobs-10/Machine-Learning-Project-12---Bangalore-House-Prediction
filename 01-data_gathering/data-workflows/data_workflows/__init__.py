from dagster import (
    AssetSelection,
    Definitions,
    ScheduleDefinition,
    define_asset_job,
    load_assets_from_modules,
    FilesystemIOManager,  # Update the imports at the top of the file to also include this
)

from . import preprocessing, feature_engineering, data_profiles

preprocessing_asset = load_assets_from_modules([preprocessing],group_name="Preprocessing")
data_profile_asset = load_assets_from_modules([data_profiles],group_name="pipelines_data_profiles")
feature_engineering_asset = load_assets_from_modules([feature_engineering],group_name="feature_engineering")
all_assets = [*preprocessing_asset, *feature_engineering_asset, *data_profile_asset]

defs = Definitions(
    assets=all_assets,
)
