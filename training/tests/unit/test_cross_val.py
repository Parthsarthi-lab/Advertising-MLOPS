import pytest
from unittest.mock import patch, Mock, mock_open
import os
import pandas as pd
import numpy as np

from training.components.cross_val.cross_val import CrossVal
from training.configuration_manager.configuration import ConfigurationManager

@pytest.fixture
def cross_val_config(tmp_path):
    """
    Fixture to create a mocked DataValidationConfig object with temporary paths.
    """
    config = Mock()
    config.root_dir = tmp_path / "cross_validation"
    config.data_dir = tmp_path / "data.csv"
    config.final_train_data_path = config.root_dir / "data_for_final_train"
    config.final_test_data_path = config.root_dir / "data_for_final_train"
    config.best_model_params = config.root_dir / "best_model_params"
    config.STATUS_FILE = config.root_dir / "status.txt"
    

    os.makedirs(config.root_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.final_train_data_path), exist_ok=True)

    # Create mock data file
    df = pd.DataFrame({
        "TV": [1.2, 2.4, 3.45],
        "radio": [1.1, 2.2, 3.3],
        "newspaper": [7.4, 8.5, 9.6],
        "sales": [1.3,14.4,5.6]
    })
    df.to_csv(config.data_dir, index=True)

    yield config

    # Cleanup
    if os.path.exists(config.data_dir):
        os.remove(config.data_dir)
    if os.path.exists(config.STATUS_FILE):
        os.remove(config.STATUS_FILE)

    if os.path.exists(config.final_train_data_path / "Train.npz"):
        os.remove(config.final_train_data_path / "Train.npz")
        
        if os.path.exists(config.final_test_data_path / "Test.npz"):
            os.remove(config.final_test_data_path / "Test.npz")

        os.rmdir(config.final_train_data_path)

    if os.path.exists(config.best_model_params):
        os.rmdir(config.best_model_params)

    os.rmdir(config.root_dir)


def test_load_ingested_data(cross_val_config):
    cross_val = CrossVal(config=cross_val_config)
    X,y = cross_val.load_ingested_data()

    assert X.shape == (3, 3)
    assert y.shape == (3,)

def test_split_data_for_final_train(cross_val_config):
    cross_val = CrossVal(config=cross_val_config)
    X,y = cross_val.load_ingested_data()

    xtrain, xtest, ytrain, ytest = cross_val.split_data_for_final_train(X,y)

    assert xtrain.shape == (2, 3)
    assert xtest.shape == (1, 3)
    assert ytrain.shape == (2,)
    assert ytest.shape == (1,)

@patch("numpy.savez")
def test_save_data_for_final_train(mock_savez,cross_val_config):
    cross_val = CrossVal(config=cross_val_config)
    X,y = cross_val.load_ingested_data()

    xtrain, xtest, ytrain, ytest = cross_val.split_data_for_final_train(X,y)

    cross_val.save_data_for_final_train(xtrain, xtest, ytrain, ytest)

    assert mock_savez.call_count == 2
     # Assert the first call arguments
    mock_savez.assert_any_call(
        os.path.join(cross_val_config.final_train_data_path, "Train.npz"),
        xtrain=xtrain,
        ytrain=ytrain,
    )
    # Assert the second call arguments
    mock_savez.assert_any_call(
        os.path.join(cross_val_config.final_test_data_path, "Test.npz"),
        xtest=xtest,
        ytest=ytest,
    )



@patch("training.components.cross_val.cross_val.GridSearchCV")
@patch("training.components.cross_val.cross_val.open", new_callable=mock_open)  # Mock file handling
@patch("training.components.cross_val.cross_val.json.dump")  # Mock JSON dumping
def test_run_cross_val(mock_json_dump, mock_open_file, mock_grid_search, cross_val_config):
    # Create mock data
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y = pd.Series([10, 20, 30])

    # Mock GridSearchCV and its behavior
    mock_gs_instance = mock_grid_search.return_value
    mock_gs_instance.best_estimator_ = Mock()
    mock_gs_instance.best_params_ = {"regressor__fit_intercept": True}
    mock_gs_instance.best_score_ = 0.95
    mock_gs_instance.best_estimator_.get_params.return_value = {"param1": 1, "param2": 2}

    # Create CrossVal instance
    cross_val = CrossVal(config=cross_val_config)

    # Call the method under test
    cross_val.run_cross_val(X, y)

    # Assertions for GridSearchCV
    mock_grid_search.assert_called_once()
    mock_gs_instance.fit.assert_called_once_with(X, y)

    # Assertions for status file
    mock_open_file.assert_any_call(cross_val_config.STATUS_FILE, "a")
    mock_open_file().write.assert_any_call("Best params for Model: {'regressor__fit_intercept': True}\n")
    mock_open_file().write.assert_any_call("Best scoring(R2) for Model: 0.95\n")

    # Assertions for saving best params
    best_params_path = os.path.join(cross_val_config.best_model_params, "best_params.json")
    mock_json_dump.assert_called_once_with({"param1": 1, "param2": 2}, mock_open_file(), indent=4)