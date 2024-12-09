import sys
import os
from pathlib import Path

from training.exception import ModelTrainingError, handle_exception
from training.custom_logging import info_logger, error_logger

from training.entity.config_entity import ModelTrainerConfig
from training.configuration_manager.configuration import ConfigurationManager

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_transformed_data(self):
        try:
            info_logger.info("Loading Final Training Transformed Data")

            info_logger.info("Loaded Final Training Transformed Data")
        except Exception as e:
            handle_exception(e, ModelTrainingError)

    def train_model(self, xtrain, xtest, ytrain, ytest):
        try:
            info_logger.info("Training final model started")

            info_logger.info("Final model trained")
        except Exception as e:
            handle_exception(e, ModelTrainingError)

    def save_model(self, model):
        try:
            info_logger.info("Saving final model started")

            info_logger.info("Final model saved")
        except Exception as e:
            handle_exception(e, ModelTrainingError)

if __name__ == "__main__":
    config = ConfigurationManager()
    model_trainer_config = config.get_model_trainer_config()

    model_trainer = ModelTrainer(config = model_trainer_config)
    xtrain, xtest, ytrain, ytest = model_trainer.load_transformed_data()
    model = model_trainer.train_model(xtrain, xtest, ytrain, ytest)
    model_trainer.save_model(model)