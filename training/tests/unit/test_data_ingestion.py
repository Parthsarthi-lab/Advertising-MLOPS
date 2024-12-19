import pytest
from unittest.mock import Mock, patch
import os
from training.configuration_manager.configuration import ConfigurationManager
from training.components.common.data_ingestion import DataIngestion



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
def config_manager(data_ingestion_config):
    config_manager = Mock()
    config_manager.get_data_ingestion_config.return_value = data_ingestion_config
    return config_manager


def test_initialization(data_ingestion_config, config_manager):
    data_ingestion_configuration = config_manager.get_data_ingestion_config()
    data_ingestion = DataIngestion(config = data_ingestion_configuration)

    assert data_ingestion.config == data_ingestion_config


def test_save_data_copy_file( data_ingestion_config, config_manager):
    
    data_ingestion_configuration = config_manager.get_data_ingestion_config()
    data_ingestion  = DataIngestion(config = data_ingestion_configuration)

    data_ingestion.save_data()

    
    assert os.path.exists(data_ingestion_config.data_dir / "advertising_data.csv")

    with open(data_ingestion_config.STATUS_FILE, "r") as f:
        status = f.read()

    assert status == "Data Ingestion status: True"

@patch("os.path.exists", return_value=True)
def test_save_data_file_exists(mock_exists,data_ingestion_config, config_manager):
    
    data_ingestion_configuration = config_manager.get_data_ingestion_config()
    data_ingestion  = DataIngestion(config = data_ingestion_configuration)

    data_ingestion.save_data()


    with open(data_ingestion_config.STATUS_FILE, "r") as f:
        status = f.read()

    assert status == "Data Ingestion status: True"

@patch("shutil.copy")
@patch("os.path.exists", side_effect=FileNotFoundError)  # side_effect not side_effects
def test_save_data_copy_exception(mock_exists,mock_copy, config_manager):
    data_ingestion_configuration = config_manager.get_data_ingestion_config()
    data_ingestion  = DataIngestion(config = data_ingestion_configuration)

    data_ingestion.save_data()
    
    with open(data_ingestion.config.STATUS_FILE, "r") as f:
        status = f.read()

    assert status == "Data Ingestion status: False"

