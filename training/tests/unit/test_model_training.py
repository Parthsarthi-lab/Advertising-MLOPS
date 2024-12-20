import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import numpy as np
import json
import os
from training.components.final_train.model_training import ModelTrainer
from training.entity.config_entity import ModelTrainerConfig

@pytest.fixture
def model_trainer_config(tmp_path):
    """
    Fixture to create a mocked ModelTrainerConfig object with temporary paths.
    """
    config = Mock()
    config.root_dir = tmp_path / "model_training"
    config.final_train_data_path = config.root_dir / "final_train"
    config.final_test_data_path = config.root_dir / "final_test"
    config.best_model_params = config.root_dir / "best_model_params"
    config.STATUS_FILE = config.root_dir / "status.txt"

    # Create directories
    os.makedirs(config.final_train_data_path, exist_ok=True)
    os.makedirs(config.final_test_data_path, exist_ok=True)
    os.makedirs(config.best_model_params, exist_ok=True)

    # Create mock data files
    np.savez(config.final_train_data_path / "Train.npz", xtrain=np.array([[1, 2], [3, 4]]), ytrain=np.array([10, 20]))
    np.savez(config.final_test_data_path / "Test.npz", xtest=np.array([[5, 6]]), ytest=np.array([30]))

    # Create mock hyperparameters file
    with open(config.best_model_params / "best_params.json", "w") as f:
        json.dump({"fit_intercept": True}, f)

    yield config

    # Cleanup
    if os.path.exists(config.STATUS_FILE):
        os.remove(config.STATUS_FILE)
    if os.path.exists(config.final_train_data_path / "Train.npz"):
        os.remove(config.final_train_data_path / "Train.npz")
    if os.path.exists(config.final_test_data_path / "Test.npz"):
        os.remove(config.final_test_data_path / "Test.npz")
    if os.path.exists(config.best_model_params / "best_params.json"):
        os.remove(config.best_model_params / "best_params.json")

    os.rmdir(config.final_train_data_path)
    os.rmdir(config.final_test_data_path)
    os.rmdir(config.best_model_params)
    os.rmdir(config.root_dir)

@pytest.fixture
def model_trainer(model_trainer_config):
    return ModelTrainer(config=model_trainer_config)

# Test load_transformed_data
def test_load_transformed_data(model_trainer):
    xtrain, xtest, ytrain, ytest = model_trainer.load_transformed_data()

    # Assertions
    np.testing.assert_array_equal(xtrain, np.array([[1, 2], [3, 4]]))
    np.testing.assert_array_equal(xtest, np.array([[5, 6]]))
    np.testing.assert_array_equal(ytrain, np.array([10, 20]))
    np.testing.assert_array_equal(ytest, np.array([30]))

# Test train_model
@patch("training.components.final_train.model_training.LinearRegression")
def test_train_model(mock_linear_regression, model_trainer):
    xtrain, xtest, ytrain, ytest = model_trainer.load_transformed_data()

    
    # Mock LinearRegression
    mock_model = mock_linear_regression.return_value

    # Call the method under test
    final_model = model_trainer.train_model(xtrain, xtest, ytrain, ytest)

    # Assertions
    mock_linear_regression.assert_called_once_with()  # Ensure correct parameters passed
    mock_model.fit.assert_called_once_with(xtrain, ytrain)  # Ensure training was performed


# Test save_model
@patch("joblib.dump")
def test_save_model(mock_joblib_dump, model_trainer):
    mock_model = Mock()

    # Call the method under test
    model_trainer.save_model(mock_model)

    # Assert model saving
    mock_joblib_dump.assert_called_once_with(
        mock_model, os.path.join(model_trainer.config.root_dir, "final_model.joblib")
    )

    # Assert status file content
    with open(model_trainer.config.STATUS_FILE, "r") as f:
        assert f.read() == "Model Training status: True"
