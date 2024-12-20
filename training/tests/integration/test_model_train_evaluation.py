import os
import pytest
import numpy as np
import json
import joblib
from unittest.mock import Mock
from training.components.final_train.model_training import ModelTrainer
from training.components.final_train.model_evaluation import ModelEvaluation

@pytest.fixture
def model_training_config(tmp_path):
    config = Mock()
    config.root_dir = tmp_path / "model_training"
    config.final_train_data_path = tmp_path / "feature_engineering" / "final_train"
    config.final_test_data_path = tmp_path / "feature_engineering" / "final_test"
    config.best_model_params = tmp_path / "model_training" / "best_model_params"
    config.STATUS_FILE = config.root_dir / "status.txt"

    # Create necessary directories
    os.makedirs(config.final_train_data_path, exist_ok=True)
    os.makedirs(config.final_test_data_path, exist_ok=True)
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


@pytest.fixture
def model_evaluation_config(tmp_path):
    config = Mock()
    config.model_path = tmp_path / "model_training"
    config.test_data_path = tmp_path / "feature_engineering" / "final_test"
    config.STATUS_FILE = tmp_path / "model_evaluation" / "evaluation_status.txt"

    os.makedirs(config.model_path, exist_ok=True)
    os.makedirs(tmp_path / "model_evaluation", exist_ok=True)

    yield config

    # Cleanup
    for file in os.listdir(tmp_path / "model_evaluation"):
        os.remove(os.path.join(tmp_path / "model_evaluation", file))
    os.rmdir(tmp_path / "model_evaluation")

def test_model_training_to_evaluation(model_training_config, model_evaluation_config):
    # Model Training
    model_trainer = ModelTrainer(config=model_training_config)

    # Prepare mock data
    xtrain = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ytrain = np.array([1, 2, 3])
    xtest = np.array([[7.0, 8.0], [9.0, 10.0]])
    ytest = np.array([4, 5])

    # Ensure directories exist before saving
    os.makedirs(model_training_config.final_train_data_path, exist_ok=True)
    os.makedirs(model_training_config.final_test_data_path, exist_ok=True)

    # Save transformed data
    np.savez(os.path.join(model_training_config.final_train_data_path, "Train.npz"), xtrain=xtrain, ytrain=ytrain)
    np.savez(os.path.join(model_training_config.final_test_data_path, "Test.npz"), xtest=xtest, ytest=ytest)

    # Train model
    model = model_trainer.train_model(xtrain, xtest, ytrain, ytest)
    assert model is not None

    # Save model
    model_trainer.save_model(model)
    model_path = os.path.join(model_training_config.root_dir, "final_model.joblib")
    assert os.path.exists(model_path)

    # Model Evaluation
    model_evaluation = ModelEvaluation(config=model_evaluation_config)
    final_model = model_evaluation.load_model()
    assert final_model is not None

    xtest_loaded, ytest_loaded = model_evaluation.load_test_data()
    assert np.array_equal(xtest, xtest_loaded)
    assert np.array_equal(ytest, ytest_loaded)

    # Evaluate model
    model_evaluation.evaluate_model(final_model, xtest_loaded, ytest_loaded)

    # Verify evaluation status file
    with open(model_evaluation_config.STATUS_FILE, "r") as f:
        content = f.read()
        assert "Model Evaluation status: True" in content
        assert "RMSE" in content
        assert "R2" in content
