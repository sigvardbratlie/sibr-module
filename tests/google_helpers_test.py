import os
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from dotenv import load_dotenv
os.chdir("..")
load_dotenv()

# Importer klassene dine
from src.sibr_module import *

# Hent test-variabler fra miljøet
PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")
SECRET_ID = os.getenv("SECRET_ID_TEST")


@pytest.fixture
def mock_gcp_clients(mocker):
    """En pytest fixture som mocker alle GCP klienter."""
    mocks = {
        'bigquery': mocker.patch('sibr_module.google_helpers.bigquery.Client', autospec=True),
        'storage': mocker.patch('sibr_module.google_helpers.storage.Client', autospec=True),
        'secretmanager': mocker.patch('sibr_module.google_helpers.secretmanager.SecretManagerServiceClient',
                                      autospec=True),
        'logging': mocker.patch('sibr_module.google_helpers.cloud_logging.Client', autospec=True)
    }
    return mocks


class TestLogger:
    def test_logger_initialization_local(self, tmp_path):
        """Tester at loggeren kan startes lokalt og lager en loggfil."""
        # Bruk en midlertidig mappe for loggfilen
        os.chdir(tmp_path)
        log = Logger("test_log")
        log.info("Dette er en test.")

        log_file = tmp_path / 'logfiles' / 'test_log.log'
        assert log_file.exists()
        assert "Dette er en test" in log_file.read_text()

    def test_logger_cloud_logging_setup(self, mock_gcp_clients):
        """Tester at Cloud Logging-oppsettet blir kalt."""
        log = Logger("cloud_test_log", enable_cloud_logging=True)
        # Sjekk at logging.Client() ble kalt
        mock_gcp_clients['logging'].assert_called_once()
        # Sjekk at setup_logging() ble kalt på instansen
        mock_gcp_clients['logging'].return_value.setup_logging.assert_called_once()


class TestBigQuery:
    def test_bq_initialization(self, mock_gcp_clients):
        """Tester at BigQuery-klienten blir initialisert korrekt."""
        bq = BigQuery(project_id=PROJECT_ID)
        mock_gcp_clients['bigquery'].assert_called_once_with(project=PROJECT_ID)
        assert bq.project == PROJECT_ID

    def test_to_bq_append(self, mock_gcp_clients):
        """Tester at to_bq kaller riktige metoder for en 'append' operasjon."""
        bq = BigQuery(project_id=PROJECT_ID)
        df = pd.DataFrame({'a': [1, 2]})

        # Kjør funksjonen
        bq.to_bq(df, "test_table", "test_dataset", if_exists="append")

        # Sjekk at load_table_from_dataframe ble kalt
        mock_gcp_clients['bigquery'].return_value.load_table_from_dataframe.assert_called_once()

    def test_get_dtype(self):
        bq = BigQuery(project_id=PROJECT_ID)
        df = pd.DataFrame({'a': [1,2],
              "b": ["hei","du"],
              "c" : [[2,3,4], ["2","3","4"]],})
        dtype_a = bq._get_dtype(df, 'a')
        dtype_b = bq._get_dtype(df, 'b')
        dtype_c = bq._get_dtype(df, 'c')
        print(dtype_a, dtype_b, dtype_c)

        assert dtype_a == 'int64'
        assert dtype_b == 'str'
        assert dtype_c == 'list'


class TestCStorage:
    def test_cstorage_initialization(self, mock_gcp_clients):
        """Tester at CStorage-klienten blir initialisert korrekt."""
        cs = CStorage(project_id=PROJECT_ID, bucket_name=BUCKET_NAME)
        mock_gcp_clients['storage'].assert_called_once_with(project=PROJECT_ID)
        assert cs._bucket_name == BUCKET_NAME

    def test_upload(self, mock_gcp_clients, tmp_path):
        """Tester at upload kaller riktige metoder."""
        cs = CStorage(project_id=PROJECT_ID, bucket_name=BUCKET_NAME)

        # Lag en falsk lokal fil
        local_file = tmp_path / "test.txt"
        local_file.write_text("hello world")

        # Mock bucket- og blob-objektene
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_gcp_clients['storage'].return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        cs.upload(str(local_file), "remote/test.txt")

        # Sjekk at metodene ble kalt med riktige argumenter
        mock_gcp_clients['storage'].return_value.bucket.assert_called_with(BUCKET_NAME)
        mock_bucket.blob.assert_called_with("remote/test.txt")
        mock_blob.upload_from_filename.assert_called_with(str(local_file))


class TestSecretsManager:
    def test_get_secret(self, mock_gcp_clients):
        """Tester at get_secret kaller riktige metoder."""
        sm = SecretsManager(project_id=PROJECT_ID)

        # Definer hva den mockede metoden skal returnere
        mock_response = MagicMock()
        mock_response.payload.data.decode.return_value = "superhemmelig"
        mock_gcp_clients['secretmanager'].return_value.access_secret_version.return_value = mock_response

        secret_value = sm.get_secret(SECRET_ID)

        # Forventet format på secret-navnet
        expected_name = f'projects/{PROJECT_ID}/secrets/{SECRET_ID}/versions/latest'

        # Sjekk at den ble kalt med riktig navn
        mock_gcp_clients['secretmanager'].return_value.access_secret_version.assert_called_with({'name': expected_name})
        assert secret_value == "superhemmelig"