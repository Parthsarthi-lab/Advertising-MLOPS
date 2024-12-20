import pytest
from unittest.mock import Mock, patch, MagicMock, ANY
import numpy as np
import os
from training.components.final_train.feature_engineering import FeatureEngineering
from training.entity.config_entity import FeatureEngineeringConfig

@pytest.fixture
def feature_engineering_config(tmp_path):
    """
    Fixture to create a mocked FeatureEngineeringConfig object with temporary paths.
    """
    config = Mock()
    config.root_dir = tmp_path / "feature_engineering"
    config.final_train_data_path = config.root_dir / "final_train"
    config.final_test_data_path = config.root_dir / "final_test"
    config.STATUS_FILE = config.root_dir / "status.txt"

    # Create directories
    os.makedirs(config.final_train_data_path, exist_ok=True)
    os.makedirs(config.final_test_data_path, exist_ok=True)

    # Create mock data files
    np.savez(config.final_train_data_path / "Train.npz", xtrain=np.array([[1, 2], [3, 4]]), ytrain=np.array([10, 20]))
    np.savez(config.final_test_data_path / "Test.npz", xtest=np.array([[5, 6]]), ytest=np.array([30]))

    yield config

    # Cleanup
    if os.path.exists(config.STATUS_FILE):
        os.remove(config.STATUS_FILE)
    if os.path.exists(config.final_train_data_path / "Train.npz"):
        os.remove(config.final_train_data_path / "Train.npz")
    if os.path.exists(config.final_test_data_path / "Test.npz"):
        os.remove(config.final_test_data_path / "Test.npz")

    os.rmdir(config.final_train_data_path)
    os.rmdir(config.final_test_data_path)
    os.rmdir(config.root_dir)

@pytest.fixture
def feature_engineering(feature_engineering_config):
    return FeatureEngineering(config=feature_engineering_config)

# Test load_saved_data
def test_load_saved_data(feature_engineering):
    xtrain, xtest, ytrain, ytest = feature_engineering.load_saved_data()

    # Assertions
    np.testing.assert_array_equal(xtrain, np.array([[1, 2], [3, 4]]))
    np.testing.assert_array_equal(xtest, np.array([[5, 6]]))
    np.testing.assert_array_equal(ytrain, np.array([10, 20]))
    np.testing.assert_array_equal(ytest, np.array([30]))

# Test transform_data
@patch("joblib.dump")
def test_transform_data(mock_joblib_dump, feature_engineering):
    xtrain, xtest, ytrain, ytest = feature_engineering.load_saved_data()

    # Call the method
    transformed_xtrain, transformed_xtest, transformed_ytrain, transformed_ytest = feature_engineering.transform_data(
        xtrain, xtest, ytrain, ytest
    )

    # Assertions
    assert transformed_xtrain.shape == xtrain.shape
    assert transformed_xtest.shape == xtest.shape

    # Assert pipeline saving
    mock_joblib_dump.assert_called_once_with(
        ANY, os.path.join(feature_engineering.config.root_dir, "pipeline.joblib")
    )

# Test save_transformed_data
@patch("numpy.savez")
def test_save_transformed_data(mock_savez, feature_engineering):
    # Mock transformed data
    transformed_xtrain = np.array([[0.1, -0.2], [1.5, 0.3]])
    transformed_xtest = np.array([[0.5, 0.8]])
    transformed_ytrain = np.array([10, 20])
    transformed_ytest = np.array([30])

    # Call the method under test
    feature_engineering.save_transformed_data(
        transformed_xtrain, transformed_xtest, transformed_ytrain, transformed_ytest
    )

    # Assert save calls
    mock_savez.assert_any_call(
        os.path.join(feature_engineering.config.root_dir, "Train.npz"),
        xtrain=transformed_xtrain,
        ytrain=transformed_ytrain
    )
    mock_savez.assert_any_call(
        os.path.join(feature_engineering.config.root_dir, "Test.npz"),
        xtest=transformed_xtest,
        ytest=transformed_ytest
    )

    # Assert status file content
    with open(feature_engineering.config.STATUS_FILE, "r") as f:
        assert f.read() == "Feature Engineering status: True"
