import os
import pytest
import numpy as np
import json
from unittest.mock import Mock
from training.components.final_train.feature_engineering import FeatureEngineering
from training.components.final_train.model_training import ModelTrainer

@pytest.fixture
def feature_engineering_config(tmp_path):
    config = Mock()
    config.root_dir = tmp_path / "feature_engineering"
    config.final_train_data_path = tmp_path / "feature_engineering" / "final_train"
    config.final_test_data_path = tmp_path / "feature_engineering" / "final_test"
    config.STATUS_FILE = config.root_dir / "status.txt"

    os.makedirs(config.root_dir, exist_ok=True)
    os.makedirs(config.final_train_data_path, exist_ok=True)
    os.makedirs(config.final_test_data_path, exist_ok=True)

    # Mock data
    xtrain = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ytrain = np.array([1, 2, 3])
    xtest = np.array([[7.0, 8.0], [9.0, 10.0]])
    ytest = np.array([4, 5])

    np.savez(os.path.join(config.final_train_data_path, "Train.npz"), xtrain=xtrain, ytrain=ytrain)
    np.savez(os.path.join(config.final_test_data_path, "Test.npz"), xtest=xtest, ytest=ytest)

    yield config

    # Cleanup
    for file in os.listdir(config.final_train_data_path):
        os.remove(os.path.join(config.final_train_data_path, file))
    os.rmdir(config.final_train_data_path)

    for file in os.listdir(config.final_test_data_path):
        os.remove(os.path.join(config.final_test_data_path, file))
    os.rmdir(config.final_test_data_path)

    for file in os.listdir(config.root_dir):
        os.remove(os.path.join(config.root_dir, file))
    os.rmdir(config.root_dir)


@pytest.fixture
def model_training_config(tmp_path):
    config = Mock()
    config.root_dir = tmp_path / "model_training"
    config.final_train_data_path = tmp_path / "feature_engineering" / "final_train"
    config.final_test_data_path = tmp_path / "feature_engineering" / "final_test"
    config.best_model_params = tmp_path / "model_training" / "best_model_params"
    config.STATUS_FILE = config.root_dir / "status.txt"

    os.makedirs(config.root_dir, exist_ok=True)
    os.makedirs(config.best_model_params, exist_ok=True)

    # Mock hyperparameters file
    hyperparams = {"regressor__fit_intercept": True}
    with open(os.path.join(config.best_model_params, "best_params.json"), "w") as f:
        json.dump(hyperparams, f)

    yield config

    # Cleanup
    for file in os.listdir(config.best_model_params):
        os.remove(os.path.join(config.best_model_params, file))
    os.rmdir(config.best_model_params)

    for file in os.listdir(config.root_dir):
        os.remove(os.path.join(config.root_dir, file))
    os.rmdir(config.root_dir)

def test_feature_engineering_to_model_training(feature_engineering_config, model_training_config):
    feature_engineering = FeatureEngineering(config=feature_engineering_config)
    model_trainer = ModelTrainer(config=model_training_config)

    # Step 1: Feature Engineering
    xtrain, xtest, ytrain, ytest = feature_engineering.load_saved_data()
    xtrain, xtest, ytrain, ytest = feature_engineering.transform_data(xtrain, xtest, ytrain, ytest)
    feature_engineering.save_transformed_data(xtrain, xtest, ytrain, ytest)

    # Verify transformed data exists
    transformed_train_path = os.path.join(feature_engineering_config.root_dir, "Train.npz")
    transformed_test_path = os.path.join(feature_engineering_config.root_dir, "Test.npz")
    assert os.path.exists(transformed_train_path)
    assert os.path.exists(transformed_test_path)

    transformed_train_data = np.load(transformed_train_path, allow_pickle=True)
    transformed_test_data = np.load(transformed_test_path, allow_pickle=True)
    assert transformed_train_data["xtrain"].shape == xtrain.shape
    assert transformed_test_data["xtest"].shape == xtest.shape

    # Step 2: Model Training
    xtrain, xtest, ytrain, ytest = model_trainer.load_transformed_data()
    model = model_trainer.train_model(xtrain, xtest, ytrain, ytest)
    assert model is not None

    # Save the trained model
    model_trainer.save_model(model)

    # Verify model file exists
    model_path = os.path.join(model_training_config.root_dir, "final_model.joblib")
    assert os.path.exists(model_path)

    # Verify status file exists
    with open(model_training_config.STATUS_FILE, "r") as f:
        status = f.read()
        assert status == "Model Training status: True"
