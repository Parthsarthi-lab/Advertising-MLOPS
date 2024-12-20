import pytest
import os
import pandas as pd
import numpy as np
import json
from unittest.mock import Mock
from training.components.common.data_validation import DataValidation
from training.components.cross_val.cross_val import CrossVal

@pytest.fixture
def data_validation_config(tmp_path):
    config = Mock()
    config.root_dir = tmp_path / "data_validation"
    config.data_dir = config.root_dir / "data.csv"
    config.STATUS_FILE = config.root_dir / "status.txt"
    config.all_schema = {
        "TV": "float64",
        "radio": "float64",
        "newspaper": "float64",
        "sales": "float64"
    }

    os.makedirs(config.root_dir, exist_ok=True)

    # Create mock data file
    df = pd.DataFrame({
        "TV": [1.2, 2.4, 3.45],
        "radio": [1.1, 2.2, 3.3],
        "newspaper": [7.4, 8.5, 9.6],
        "sales": [1.3, 14.4, 5.6]
    })
    df.to_csv(config.data_dir, index=True)

    yield config

    # Cleanup
    if os.path.exists(config.data_dir):
        os.remove(config.data_dir)
    if os.path.exists(config.STATUS_FILE):
        os.remove(config.STATUS_FILE)
    os.rmdir(config.root_dir)

@pytest.fixture
def cross_val_config(tmp_path):
    config = Mock()
    config.root_dir = tmp_path / "cross_validation"
    config.data_dir = tmp_path / "data.csv"
    config.final_train_data_path = config.root_dir / "data_for_final_train"
    config.final_test_data_path = config.root_dir / "data_for_final_test"
    config.best_model_params = config.root_dir / "best_model_params"
    config.STATUS_FILE = config.root_dir / "status.txt"

    os.makedirs(config.root_dir, exist_ok=True)
    os.makedirs(config.final_train_data_path, exist_ok=True)
    os.makedirs(config.final_test_data_path, exist_ok=True)
    os.makedirs(config.best_model_params, exist_ok=True)

    # Create mock data file
    df = pd.DataFrame({
        "TV": [1.2, 2.4, 3.45, 45.7, 34.23, 78.6],
        "radio": [1.1, 2.2, 3.3, 12, 2, 42.8],
        "newspaper": [7.4, 8.5, 9.6, 1.45, 2.54, 3.6],
        "sales": [1.3, 14.4, 5.6, 2.3, 3.4, 4.5]
    })
    df.to_csv(config.data_dir, index=True)

    yield config

    # Cleanup
    if os.path.exists(config.data_dir):
        os.remove(config.data_dir)
    if os.path.exists(config.STATUS_FILE):
        os.remove(config.STATUS_FILE)
    for file in os.listdir(config.final_train_data_path):
        os.remove(os.path.join(config.final_train_data_path, file))
    for file in os.listdir(config.final_test_data_path):
        os.remove(os.path.join(config.final_test_data_path, file))
    for file in os.listdir(config.best_model_params):
        os.remove(os.path.join(config.best_model_params, file))
    os.rmdir(config.final_train_data_path)
    os.rmdir(config.final_test_data_path)
    os.rmdir(config.best_model_params)
    os.rmdir(config.root_dir)

def test_data_validation_to_cross_validation(data_validation_config, cross_val_config):
    data_validation = DataValidation(config=data_validation_config)
    cross_val = CrossVal(config=cross_val_config)

    data_validation.validate_data()
    with open(data_validation_config.STATUS_FILE, "r") as f:
        status = f.read()
        assert status == "Data Validation status: True"

    X, y = cross_val.load_ingested_data()
    assert X.shape == (6, 3)
    assert y.shape == (6,)

    xtrain, xtest, ytrain, ytest = cross_val.split_data_for_final_train(X, y)
    assert xtrain.shape[0] == 4
    assert xtest.shape[0] == 2

    cross_val.save_data_for_final_train(xtrain, xtest, ytrain, ytest)
    assert os.path.exists(os.path.join(cross_val_config.final_train_data_path, "Train.npz"))
    assert os.path.exists(os.path.join(cross_val_config.final_test_data_path, "Test.npz"))

    cross_val.run_cross_val(X, y)
    with open(cross_val_config.STATUS_FILE, "r") as f:
        content = f.read()
        assert "Best params for Model:" in content
        assert "Best scoring(R2) for Model:" in content

    best_params_path = os.path.join(cross_val_config.best_model_params, "best_params.json")
    assert os.path.exists(best_params_path)
    with open(best_params_path, "r") as f:
        params = json.load(f)
        assert any("fit_intercept" in key for key in params)
