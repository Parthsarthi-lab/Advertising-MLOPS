import os
import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import Mock
from training.components.common.data_ingestion import DataIngestion
from training.components.common.data_validation import DataValidation
from training.components.cross_val.cross_val import CrossVal
from training.components.final_train.feature_engineering import FeatureEngineering
from training.components.final_train.model_training import ModelTrainer
from training.components.final_train.model_evaluation import ModelEvaluation

@pytest.fixture
def integration_config(tmp_path):
    config = {
        "data_ingestion": Mock(),
        "data_validation": Mock(),
        "cross_val": Mock(),
        "feature_engineering": Mock(),
        "model_training": Mock(),
        "model_evaluation": Mock(),
    }

    # Data Ingestion Config
    config["data_ingestion"].root_dir = tmp_path / "data_ingestion"
    config["data_ingestion"].source = tmp_path / "data/data.csv"
    config["data_ingestion"].data_dir = config["data_ingestion"].root_dir / "data.csv"
    config["data_ingestion"].STATUS_FILE = config["data_ingestion"].root_dir / "status.txt"
    os.makedirs(config["data_ingestion"].root_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config["data_ingestion"].source), exist_ok=True)
    

    # Data Validation Config
    config["data_validation"].root_dir = tmp_path / "data_validation"
    config["data_validation"].data_dir = config["data_ingestion"].data_dir
    config["data_validation"].STATUS_FILE = config["data_validation"].root_dir / "status.txt"
    config["data_validation"].all_schema = {
        "TV": "float64",
        "radio": "float64",
        "newspaper": "float64",
        "sales": "float64",
    }
    os.makedirs(config["data_validation"].root_dir, exist_ok=True)

    # Cross Validation Config
    config["cross_val"].root_dir = tmp_path / "cross_validation"
    config["cross_val"].data_dir = config["data_ingestion"].data_dir
    config["cross_val"].final_train_data_path = config["cross_val"].root_dir / "final_train"
    config["cross_val"].final_test_data_path = config["cross_val"].root_dir / "final_test"
    config["cross_val"].best_model_params = config["cross_val"].root_dir / "best_model_params"
    config["cross_val"].STATUS_FILE = config["cross_val"].root_dir / "status.txt"
    os.makedirs(config["cross_val"].final_train_data_path, exist_ok=True)
    os.makedirs(config["cross_val"].final_test_data_path, exist_ok=True)
    os.makedirs(config["cross_val"].best_model_params, exist_ok=True)

    # Feature Engineering Config
    config["feature_engineering"].root_dir = tmp_path / "feature_engineering"
    config["feature_engineering"].final_train_data_path = config["cross_val"].final_train_data_path
    config["feature_engineering"].final_test_data_path = config["cross_val"].final_test_data_path
    config["feature_engineering"].STATUS_FILE = config["feature_engineering"].root_dir / "status.txt"
    os.makedirs(config["feature_engineering"].root_dir, exist_ok=True)

    # Model Training Config
    config["model_training"].root_dir = tmp_path / "model_training"
    config["model_training"].final_train_data_path = config["feature_engineering"].final_train_data_path
    config["model_training"].final_test_data_path = config["feature_engineering"].final_test_data_path
    config["model_training"].best_model_params = config["cross_val"].best_model_params
    config["model_training"].STATUS_FILE = config["model_training"].root_dir / "status.txt"
    os.makedirs(config["model_training"].root_dir, exist_ok=True)

    # Model Evaluation Config
    config["model_evaluation"].model_path = config["model_training"].root_dir
    config["model_evaluation"].test_data_path = config["feature_engineering"].final_test_data_path
    config["model_evaluation"].STATUS_FILE = tmp_path / "model_evaluation" / "evaluation_status.txt"
    os.makedirs(tmp_path / "model_evaluation", exist_ok=True)

    # Generate mock raw data for data ingestion
    df = pd.DataFrame({
        "TV": [1.2, 2.4, 3.45, 45.7, 34.23, 78.6],
        "radio": [1.1, 2.2, 3.3, 12, 2, 42.8],
        "newspaper": [7.4, 8.5, 9.6, 1.45, 2.54, 3.6],
        "sales": [1.3, 14.4, 5.6, 2.3, 3.4, 4.5]
    })
    df.to_csv(config["data_ingestion"].source, index=True)

    yield config

    # Cleanup
    for component in config.values():
        # Ensure the component has a valid root_dir path
        if hasattr(component, "root_dir") and isinstance(component.root_dir, (str, os.PathLike)):
            for root, _, files in os.walk(component.root_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                os.rmdir(root)


def test_full_pipeline(integration_config):
    # Data Ingestion
    data_ingestion = DataIngestion(config=integration_config["data_ingestion"])
    data_ingestion.save_data()
    assert os.path.exists(integration_config["data_ingestion"].STATUS_FILE)

    # Data Validation
    data_validation = DataValidation(config=integration_config["data_validation"])
    data_validation.validate_data()
    with open(integration_config["data_validation"].STATUS_FILE, "r") as f:
        status = f.read()
        assert "Data Validation status: True" in status

    # Cross Validation
    cross_val = CrossVal(config=integration_config["cross_val"])
    X, y = cross_val.load_ingested_data()
    xtrain, xtest, ytrain, ytest = cross_val.split_data_for_final_train(X, y)
    cross_val.save_data_for_final_train(xtrain, xtest, ytrain, ytest)
    cross_val.run_cross_val(X, y)
    assert os.path.exists(integration_config["cross_val"].STATUS_FILE)

    # Feature Engineering
    feature_engineering = FeatureEngineering(config=integration_config["feature_engineering"])
    xtrain, xtest, ytrain, ytest = feature_engineering.load_saved_data()
    xtrain, xtest, ytrain, ytest = feature_engineering.transform_data(xtrain, xtest, ytrain, ytest)
    feature_engineering.save_transformed_data(xtrain, xtest, ytrain, ytest)
    assert os.path.exists(integration_config["feature_engineering"].STATUS_FILE)

    # Model Training
    model_trainer = ModelTrainer(config=integration_config["model_training"])
    model = model_trainer.train_model(xtrain, xtest, ytrain, ytest)
    model_trainer.save_model(model)
    assert os.path.exists(os.path.join(integration_config["model_training"].root_dir, "final_model.joblib"))

    # Model Evaluation
    model_evaluation = ModelEvaluation(config=integration_config["model_evaluation"])
    final_model = model_evaluation.load_model()
    xtest, ytest = model_evaluation.load_test_data()
    model_evaluation.evaluate_model(final_model, xtest, ytest)
    with open(integration_config["model_evaluation"].STATUS_FILE, "r") as f:
        content = f.read()
        assert "Model Evaluation status: True" in content
        assert "RMSE" in content
        assert "R2" in content
