import os
os.chdir("..")
import pytest
from src.sibr_module.google_helpers import *
from dotenv import load_dotenv
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")


@pytest.fixture(scope="module", autouse=True)
def test_fix():
    return BigQuery(project_id=PROJECT_ID)

def test_init(test_fix):
    assert test_fix.project == PROJECT_ID, "project should be set by init"
    assert test_fix._logger is not None, "Logger should not be None"
    assert test_fix._bq_client.project == PROJECT_ID, "project should be set by init"