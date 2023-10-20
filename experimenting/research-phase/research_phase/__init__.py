from dagster import (
    AssetSelection,
    Definitions,
    ScheduleDefinition,
    define_asset_job,
    load_assets_from_modules,
    FilesystemIOManager,  # Update the imports at the top of the file to also include this
)

from . import experiment_track

experiment_track_asset = load_assets_from_modules([experiment_track], group_name="research_phase")
all_assets = experiment_track_asset

defs = Definitions(
    assets=all_assets,
)
