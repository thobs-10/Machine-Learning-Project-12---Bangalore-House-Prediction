from experimenting.steps.get_data import ingest_data
from experimenting.steps.data_validation import data_drift_detector
from experimenting.steps.model_training import train_model
from experimenting.steps.model_testing import testing_model
from experimenting.steps.model_tuning import model_tuning
from experimenting.steps.model_validation import model_validation
from experimenting.steps.model_registry import model_registry

from experimenting.training_pipeline.training_pipeline import training_pipeline

from zenml.integrations.deepchecks.visualizers import DeepchecksVisualizer
from zenml.logger import get_logger

logger = get_logger(__name__)

threshold = 0.80

def main():
    pipeline_instance = training_pipeline(
        threshold,
        ingest_data(),
        train_model(),
        testing_model(),
        # model_tuning(),
        # model_validation(),
        # model_registry(),
        data_drift_detector,
    )
    pipeline_instance.run(run_name="trainV9.3")

    last_run = pipeline_instance.get_runs()[0]
    data_drift_step = last_run.get_step(step="data_drift_detector")

    DeepchecksVisualizer().visualize(data_drift_step)
    
if __name__ == "__main__":
    main()