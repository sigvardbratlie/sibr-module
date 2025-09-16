import os
import numpy as np
os.chdir("..")
import pytest
from src.sibr_module.logging import *
from dotenv import load_dotenv
load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME")


class TestBigQuery:

    @pytest.fixture(scope="module", autouse=True)
    def test_fix(self):
        return BigQuery()

    def test_init(self,test_fix):
        assert test_fix.project == "sibr-admin", "project should be set by init"
        assert test_fix._logger is not None, "Logger should not be None"
        assert test_fix._bq_client.project == "sibr-admin", "project should be set by init"

    def test_to_bq(self, test_fix):
        df = pd.DataFrame({'a': [1, np.nan]})
        test_fix.to_bq(df, "test_table", "test_dataset", if_exists="append")

    def test_read_bq(self, test_fix):
        df = test_fix.read_bq("SELECT * FROM `test_dataset.test_table`")
        assert isinstance(df, pd.DataFrame), "Output must be a pandas dataframe"
        assert len(df) > 0, "Output must not be empty"


class TestCStorage:
    @pytest.fixture(scope="module", autouse=True)
    def test_fix(self):
        return CStorage(project_id=PROJECT_ID, bucket_name=BUCKET_NAME)

    def test_init(self, test_fix):
        assert test_fix._bucket_name == BUCKET_NAME
        assert test_fix._logger is not None, "Logger should not be None"
        assert test_fix.project == PROJECT_ID, "project should be set by init"

    def test_upload_txt(self, test_fix, tmp_path):
        local_file = tmp_path / "test.txt"
        local_file.write_text("hello world")
        test_fix.upload(str(local_file), "remote/test.txt")

    def test_upload_csv(self,test_fix):
        df = pd.DataFrame({'a': [1, 2]})
        df.to_csv('/tmp/test.csv', index=False)
        test_fix.upload(local_file_path='/tmp/test.csv', destination_blob_name='remote/test.csv')

    def test_download_txt(self, test_fix, tmp_path):
        #test_fix.upload(str(tmp_path / "test.txt"), "remote/test.txt")
        text = test_fix.download("remote/test.txt", read_in_file=True)
        assert text is not None
        assert isinstance(text,str)

    def test_download_csv(self,test_fix):
        df = test_fix.download(source_blob_name = 'remote/test.csv',
                          read_in_file = True)
        assert isinstance(df, pd.DataFrame)
        assert len(df)>0

class TestSecretManager:
    @pytest.fixture(scope="module", autouse=True)
    def test_fix(self):
        return SecretsManager(project_id=PROJECT_ID)

    def test_init(self, test_fix):
        assert test_fix.project == PROJECT_ID

    def test_get_secret(self, test_fix):
        secret = test_fix.get_secret("test_secret")
        assert secret == "Hello World!"


class TestLogger:
    @pytest.fixture(scope="module", autouse=True)
    def test_fix(self):
        # --- Oppsett-kode (kjører før testene) ---
        logger_instance = Logger("test_logger", enable_cloud_logging=True, cloud_log_name="test_gcloud_logging")

        yield logger_instance  # Her kjøres testene dine

        # --- Ryddingskode (kjører etter at alle testene i modulen er ferdige) ---
        print("\nShutting down logger manually...")
        logger_instance.shutdown()

    def test_init(self, test_fix):
        assert test_fix._logName == "test_logger"
        assert test_fix.cloud_log_name == "test_gcloud_logging"
        assert test_fix._logger is not None, "Logger should not be None"

    def test_logging_something(self, test_fix):
        test_fix.info("This is a test message from pytest.")
