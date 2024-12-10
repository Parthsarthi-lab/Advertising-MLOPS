from training.custom_logging import info_logger, error_logger
from training.exception import FeatureEngineeringError, handle_exception

import sys
import os
from pathlib import Path
import joblib

class FeatureEngineering:
    def __init__(self):
        pass

    def transform_data(self, data):
        try:
            transformation_pipeline_path = "artifacts/feature_engineering/pipeline.joblib"
            transformation_pipeline = joblib.load(transformation_pipeline_path)

            transformed_data = transformation_pipeline.transform(data)

            return transformed_data
        except Exception as e:
            handle_exception(e, FeatureEngineeringError)