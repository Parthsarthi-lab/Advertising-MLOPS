import os
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from training.components.common.data_validation import DataValidation
from training.exception import DataValidationError


@pytest.fixture
def data_validation_config(tmp_path):
    """
    Fixture to create a mocked DataValidationConfig object with temporary paths.
    """
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
    os.makedirs(os.path.dirname(config.data_dir), exist_ok=True)

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

    os.rmdir(config.root_dir)

@pytest.fixture
def config_manager(data_validation_config):
    config_manager = Mock()
    config_manager.get_data_validation_config.return_value = data_validation_config
    return config_manager

def test_validate_data_success(data_validation_config,config_manager):
    """
    Test validate_data with correct schema and data.
    """
    data_validation_configuration = config_manager.get_data_validation_config()
    data_validation = DataValidation(config=data_validation_configuration)
    data_validation.validate_data()

    with open(data_validation.config.STATUS_FILE, "r") as f:
        status = f.read()

    assert status == "Data Validation status: True"


def test_validate_data_column_name_mismatch(data_validation_config):
    """
    Test validate_data when column names do not match the schema.
    """
    # Modify the schema to simulate a mismatch
    data_validation_config.all_schema = {
        "wrong_column": "float64",
        "radio": "float64",
        "newspaper": "object",
        "sales": "float64"
    }

    data_validation = DataValidation(config=data_validation_config)
    data_validation.validate_data()

    with open(data_validation.config.STATUS_FILE, "r") as f:
        status = f.read()

    assert status == "Data Validation status: False"


def test_validate_data_dtype_mismatch(data_validation_config):
    """
    Test validate_data when data types do not match the schema.
    """
    # Modify the schema to simulate a mismatch
    data_validation_config.all_schema = {
        "TV": "object",
        "radio": "float64",
        "newspaper": "object",
        "sales": "float64"
    }

    data_validation = DataValidation(config=data_validation_config)
    data_validation.validate_data()

    with open(data_validation.config.STATUS_FILE, "r") as f:
        status = f.read()

    assert status == "Data Validation status: False"


@patch("pandas.read_csv", side_effect=FileNotFoundError("Mocked FileNotFoundError"))
def test_validate_data_file_not_found(mock_read_csv, data_validation_config):
    """
    Test validate_data when the data file is missing.
    """
    data_validation = DataValidation(config=data_validation_config)

    with patch("training.components.common.data_validation.handle_exception") as mock_handle_exception:
        data_validation.validate_data()

        # Assert exception handling was triggered
        mock_handle_exception.assert_called_once_with(
            mock_read_csv.side_effect, DataValidationError
        )

        with open(data_validation.config.STATUS_FILE, "r") as f:
            status = f.read()

        assert status == "Data Validation status: False"


@patch("pandas.read_csv", side_effect=ValueError("Mocked ValueError"))
def test_validate_data_invalid_file_format(mock_read_csv, data_validation_config):
    """
    Test validate_data when the data file has an invalid format.
    """
    data_validation = DataValidation(config=data_validation_config)

    with patch("training.components.common.data_validation.handle_exception") as mock_handle_exception:
        data_validation.validate_data()

        # Assert exception handling was triggered
        mock_handle_exception.assert_called_once_with(
            mock_read_csv.side_effect, DataValidationError
        )

        with open(data_validation.config.STATUS_FILE, "r") as f:
            status = f.read()

        assert status == "Data Validation status: False"
