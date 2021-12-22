"""
Model training script.
"""
import time
import logging

from src.config.directories import directories as dirs
from src.constants import c_INTERMEDIATE
from src.training.data import build_dataset
from src.io import save_training_output
from src.training.models import get_model
from src.train import train


logger = logging.getLogger(__name__)


def main():
    start = time.time()
    logger.info("Starting training job...")
    model = get_model()
   
    dataset_path = dirs.data_dir / c_INTERMEDIATE
    dataset = build_dataset()
    model_metrics = train(model, dataset)
    save_training_output(model_metrics, directory=dirs.raw_store_dir)
    run_duration = time.time() - start
    logger.info("Training job done...")
    logger.info(f"Took {run_duration} seconds to execute")


if __name__ == '__main__':
    main()