import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from training.components.cross_val.cross_val import CrossVal
from training.components.final_train.feature_engineering import FeatureEngineering

@pytest.fixture
def cross_val_config(tmp_path):
    config = Mock()
    config.root_dir = tmp_path / "cross_validation"
    config.data_dir = tmp_path / "data.csv"
    config.final_train_data_path = config.root_dir / "final_train"
    config.final_test_data_path = config.root_dir / "final_test"
    config.best_model_params = config.root_dir / "best_model_params"
    config.STATUS_FILE = config.root_dir / "status.txt"

    os.makedirs(config.final_train_data_path, exist_ok=True)
    os.makedirs(config.final_test_data_path, exist_ok=True)
    os.makedirs(config.best_model_params, exist_ok=True)

    df = pd.DataFrame({
        "TV": [230.1, 44.5, 17.2],
        "radio": [37.8, 39.3, 45.9],
        "newspaper": [69.2, 45.1, 69.3],
        "sales": [22.1, 10.4, 9.3]
    })
    df.to_csv(config.data_dir, index=False)

    yield config

    for file in os.listdir(config.final_train_data_path):
        os.remove(os.path.join(config.final_train_data_path, file))
    for file in os.listdir(config.final_test_data_path):
        os.remove(os.path.join(config.final_test_data_path, file))
    os.rmdir(config.final_train_data_path)
    os.rmdir(config.final_test_data_path)
    os.rmdir(config.best_model_params)
    os.rmdir(config.root_dir)

@pytest.fixture
def feature_engineering_config(tmp_path):
    config = Mock()
    config.root_dir = tmp_path / "feature_engineering"
    config.final_train_data_path = tmp_path / "cross_validation" / "final_train"
    config.final_test_data_path = tmp_path / "cross_validation" / "final_test"
    config.STATUS_FILE = config.root_dir / "status.txt"

    os.makedirs(config.root_dir, exist_ok=True)

    yield config

    for file in os.listdir(config.root_dir):
        os.remove(os.path.join(config.root_dir, file))
    os.rmdir(config.root_dir)

def test_cross_validation_to_feature_engineering(cross_val_config, feature_engineering_config):
    cross_val = CrossVal(config=cross_val_config)
    feature_engineering = FeatureEngineering(config=feature_engineering_config)

    # Cross Validation Step
    X, y = cross_val.load_ingested_data()
    xtrain, xtest, ytrain, ytest = cross_val.split_data_for_final_train(X, y)
    cross_val.save_data_for_final_train(xtrain, xtest, ytrain, ytest)

    # Verify saved data exists
    train_path = os.path.join(cross_val_config.final_train_data_path, "Train.npz")
    test_path = os.path.join(cross_val_config.final_test_data_path, "Test.npz")
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)

    # Feature Engineering Step
    xtrain, xtest, ytrain, ytest = feature_engineering.load_saved_data()
    assert xtrain.shape[1] == X.shape[1]  # Check feature dimensions
    assert xtest.shape[1] == X.shape[1]

    xtrain, xtest, ytrain, ytest = feature_engineering.transform_data(xtrain, xtest, ytrain, ytest)
    feature_engineering.save_transformed_data(xtrain, xtest, ytrain, ytest)

    # Verify transformed data exists
    transformed_train_path = os.path.join(feature_engineering_config.root_dir, "Train.npz")
    transformed_test_path = os.path.join(feature_engineering_config.root_dir, "Test.npz")
    assert os.path.exists(transformed_train_path)
    assert os.path.exists(transformed_test_path)

    # Load transformed data to verify
    transformed_train_data = np.load(transformed_train_path, allow_pickle=True)
    transformed_test_data = np.load(transformed_test_path, allow_pickle=True)

    assert transformed_train_data["xtrain"].shape == xtrain.shape
    assert transformed_test_data["xtest"].shape == xtest.shape
