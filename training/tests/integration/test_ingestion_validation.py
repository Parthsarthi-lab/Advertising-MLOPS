import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock
from training.components.common.data_ingestion import DataIngestion
from training.components.common.data_validation import DataValidation

@pytest.fixture
def data_ingestion_config(tmp_path):
    config = Mock()
    config.root_dir = tmp_path / "data_ingestion"
    config.source = tmp_path /  "data/advertising_data.csv"
    config.data_dir = config.root_dir
    config.STATUS_FILE = config.root_dir / "status.txt"

    os.makedirs(config.root_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.source), exist_ok=True)
    config.source.write_text("mock data")

    yield config


    os.remove(config.source)

    if os.path.exists(config.STATUS_FILE):
        os.remove(config.STATUS_FILE)
    
    if os.path.exists(config.data_dir / "advertising_data.csv"):
        os.remove(config.data_dir / "advertising_data.csv")
        
    os.rmdir(config.root_dir)
    os.rmdir(os.path.dirname(config.source))


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




def test_data_ingestion_to_validation(data_ingestion_config, data_validation_config):
    # Data Ingestion
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.save_data()

    # Data Validation
    data_validation = DataValidation(config=data_validation_config)
    data_validation.validate_data()

    # Assertions
    with open(data_validation.config.STATUS_FILE, "r") as f:
        assert f.read() == "Data Validation status: True"
