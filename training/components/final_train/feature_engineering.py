import os
import sys
from pathlib import Path

from training.custom_logging import info_logger, error_logger
from training.exception import FeatureEngineeringError, handle_exception

from training.entity.config_entity import FeatureEngineeringConfig
from training.configuration_manager.configuration import ConfigurationManager

class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def load_saved_data(self):
        try:
            info_logger.info("Loading Final Training Saved Data for Transformation")

            info_logger.info("Loaded Final Training Saved Data for Transformation")

            return xtrain, xtest, ytrain, ytest
        except Exception as e:
            handle_exception(e, FeatureEngineeringError)

    def transform_data(self, xtrain, xtest, ytrain, ytest):
        try:
            info_logger.info("Transforming Final Training Data")

            info_logger.info("Transformed Final Training Data")

            return xtrain, xtest, ytrain, ytest
        except Exception as e:
            handle_exception(e, FeatureEngineeringError)

    def save_transformed_data(self, xtrain, xtest, ytrain, ytest):
        try:
            info_logger.info("Saving Final Training Transformed Data")

            info_logger.info("Saved Final Training Transformed Data")
        except Exception as e:
            handle_exception(e, FeatureEngineeringError)

if __name__ == "__main__":
    config = ConfigurationManager()
    feature_engineering_config = config.get_feature_engineering_config()

    feature_engineering = FeatureEngineering(config = feature_engineering_config)
    xtrain, xtest, ytrain, ytest = feature_engineering.load_saved_data()
    xtrain, xtest, ytrain, ytest = feature_engineering.transform_data(xtrain, xtest, ytrain, ytest)
    feature_engineering.save_transformed_data(xtrain, xtest, ytrain, ytest)