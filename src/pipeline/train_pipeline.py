import sys

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline() -> float:
    """
    Runs the full training pipeline and returns the test R2 score of the best model.
    Artifacts are saved to the `artifacts/` folder.
    """
    try:
        logging.info("Starting training pipeline")

        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            train_path=train_path, test_path=test_path
        )

        trainer = ModelTrainer()
        r2 = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"Training pipeline completed. R2={r2}")
        return r2

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    score = run_training_pipeline()
    print(f"Training complete. Best model test R2: {score:.4f}")

