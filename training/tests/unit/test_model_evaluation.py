import pytest
from unittest.mock import Mock, patch
import numpy as np
import os
import joblib
from training.components.final_train.model_evaluation import ModelEvaluation
from training.entity.config_entity import ModelEvaluationConfig

@pytest.fixture
def model_evaluation_config(tmp_path):
    """
    Fixture to create a mocked ModelEvaluationConfig object with temporary paths.
    """
    config = Mock()
    config.model_path = tmp_path / "model_artifacts"
    config.test_data_path = tmp_path / "test_data"
    config.STATUS_FILE = tmp_path / "evaluation_status.txt"

    # Create directories
    os.makedirs(config.model_path, exist_ok=True)
    os.makedirs(config.test_data_path, exist_ok=True)

    # Create mock model file
    model_path = config.model_path / "final_model.joblib"
    mock_model = Mock()
    mock_model.predict.return_value = np.array([10, 20])
    mock_model.score.return_value = 0.85
   # joblib.dump(mock_model, model_path)

    # Create mock test data file
    np.savez(config.test_data_path / "Test.npz", xtest=np.array([[1, 2], [3, 4]]), ytest=np.array([10, 20]))

    yield config

    # Cleanup
    if os.path.exists(config.STATUS_FILE):
        os.remove(config.STATUS_FILE)
    if os.path.exists(config.model_path / "final_model.joblib"):
        os.remove(config.model_path / "final_model.joblib")
    if os.path.exists(config.test_data_path / "Test.npz"):
        os.remove(config.test_data_path / "Test.npz")

    os.rmdir(config.model_path)
    os.rmdir(config.test_data_path)

@pytest.fixture
def model_evaluation(model_evaluation_config):
    return ModelEvaluation(config=model_evaluation_config)

# Test load_model
@patch("joblib.load")
def test_load_model(mock_joblib_load, model_evaluation):
    mock_model = Mock()
    mock_joblib_load.return_value = mock_model

    final_model = model_evaluation.load_model()

    mock_joblib_load.assert_called_once_with(os.path.join(model_evaluation.config.model_path, "final_model.joblib"))
    assert final_model == mock_model

# Test load_test_data
def test_load_test_data(model_evaluation):
    xtest, ytest = model_evaluation.load_test_data()

    np.testing.assert_array_equal(xtest, np.array([[1, 2], [3, 4]]))
    np.testing.assert_array_equal(ytest, np.array([10, 20]))

# Test evaluate_model
@patch("numpy.load")
def test_evaluate_model(mock_np_load, model_evaluation):
    mock_model = Mock()
    mock_model.predict.return_value = np.array([10, 20])
    mock_model.score.return_value = 0.85

    # Mock the output of np.load
    mock_np_load.return_value = {
        "xtest": np.array([[1, 2], [3, 4]]),
        "ytest": np.array([10, 20])
    }

    # Call the method under test
    xtest, ytest = model_evaluation.load_test_data()
    model_evaluation.evaluate_model(mock_model, xtest, ytest)

    # Check if the STATUS_FILE was updated correctly
    with open(model_evaluation.config.STATUS_FILE, "r") as f:
        content = f.read()
        assert "Model Evaluation status: True" in content
        assert "RMSE: 0.0" in content
        assert "R2: 0.85" in content
