import sys
import os
from pathlib import Path

from training.exception import ModelEvaluationError, handle_exception
from training.custom_logging import info_logger, error_logger

from training.entity.config_entity import ModelEvaluationConfig
from training.configuration_manager.configuration import ConfigurationManager

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.conifg = config

    def load_model(self):
        try:
            info_logger.info("Loading  final model from artifacts folder")
            
            info_logger.info("Final model loaded from artifacts folder")
        except Exception as e:
            handle_exception(e, ModelEvaluationError)


    def load_test_data(self):
        try:
            info_logger.info("Loading test data for final model evaluation")

            info_logger.info("Loaded test data for final model evaluation")

        except Exception as e:
            handle_exception(e, ModelEvaluationError)

    def evaluate_model(self, final_model, test_data):
        try:
            info_logger.info("Model evaluation started")

            info_logger.info("Model evaluation completed")
        except Exception as e:
            handle_exception(e, ModelEvaluationError)


if __name__ == "__main__":
    config = ConfigurationManager()
    model_evaluation_config = config.get_model_evaluation_config()

    model_evluation = ModelEvaluation(config = model_evaluation_config)
    final_model = model_evluation.load_model()
    test_data = model_evluation.load_test_data()

    model_evluation.evaluate_model(final_model, test_data)
    
