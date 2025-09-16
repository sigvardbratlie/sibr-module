import pandas as pd
import uuid
import traceback
from typing import Literal,Optional,Any,Dict,List,TYPE_CHECKING
import logging
from pathlib import Path
import os
import json

from google.auth.exceptions import DefaultCredentialsError
from google.api_core.exceptions import GoogleAPICallError, NotFound, AlreadyExists
from dotenv import dotenv_values
try:
    from google.cloud import storage
except ImportError:
    raise ImportError(
        "CStorage krever 'google-cloud-storage'. "
        "Install it with: pip install 'sibr-module[storage]'"
    )



class CStorage:
    """Client for file operations against Google Cloud Storage (GCS).

        This class simplifies uploading and downloading files to a specific
        GCS bucket. It can also read the content of certain file types
        directly into memory (e.g., into a pandas DataFrame).
        """
    def __init__(self,bucket_name : str,project_id : str = None, logger = None):
        '''
        Initialize a Google Cloud Storage client.
        :param project_id:
        :param bucket_name:
        :param logger: Optional, a Logger instance for logging.
        '''

        if logger is None:
            logger = logging.getLogger("CStorage")
        self.logger = logger
        self._bucket_name = bucket_name
        if project_id is None:
            cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            with open(cred_path, "r") as f:
                cred = json.load(f)
                project_id = cred.get("project_id")
        self.project = project_id
        try:
            self._client = storage.Client(project=project_id)
            self.logger.info(f"Google Cloud Storage client initialized with bucket: {self._bucket_name}")
        except Exception as e:
            self.logger.error(f"Error initializing Google Cloud Storage client: {e}")
            raise ImportError(f"Error initializing Google Cloud Storage client: {e}")

    def upload(self, local_file_path : str, destination_blob_name : str=None):
        '''
        Uploads a file to Google Cloud Storage.
        :param local_file_path: Inlcudes the full path to the file with file-extension.
        :param destination_blob_name: Includes the full path to the file with file-extension.
        :return:
        '''

        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file {local_file_path} does not exist.")
        if destination_blob_name is None:
            if '/' not in local_file_path:
                destination_blob_name = local_file_path.split('/')[-1]
            else:
                destination_blob_name = local_file_path
        try:
            bucket = self._client.bucket(self._bucket_name)
            blob = bucket.blob(destination_blob_name)

            blob.upload_from_filename(local_file_path)
            self._logger.info(
                f"File {local_file_path} uploaded to {destination_blob_name} in bucket {self._bucket_name}.")
        except Exception as e:
            self._logger.error(f"Failed to upload file {local_file_path} to bucket {self._bucket_name}: {e}")
            raise e

    def download(self, source_blob_name : str, destination_file_path : str = None, read_in_file : bool =False):
        '''
        Downloads a file from Google Cloud Storage. Available formats are ['pkl', 'csv','txt','json','xlsx','xls','yaml','yml'].
        .csv and .xlsx and .xls are read into to a pandas dataframe
        :param source_blob_name:
        :param destination_file_path:
        :param read_in_file: bool
        :return: returns the file if read_in_file is True
        '''
        if not read_in_file and not destination_file_path:
            raise ValueError(
                "Either destination_file_path must be provided or read_in_file must be True to read the file content directly.")
        try:
            bucket = self._client.bucket(self._bucket_name)
            blob = bucket.blob(source_blob_name)
            name = os.path.basename(blob.name)
            temp_filepath = f'/tmp/{name}'
            if destination_file_path:
                blob.download_to_filename(destination_file_path)
                self._logger.info(f"Blob {source_blob_name} downloaded to {destination_file_path}.")
            if read_in_file:
                valid_ext = ['pkl', 'csv','txt','json','xlsx','xls','yaml','yml']
                if "." in name:
                    ext = name.split('.')[-1]
                    if ext not in valid_ext:
                        raise ValueError(f"Invalid file extension: {ext}. Valid extensions are: {valid_ext}")
                    try:
                        blob.download_to_filename(temp_filepath)
                        if ext == 'csv':
                            output = pd.read_csv(temp_filepath)
                        elif ext in ["xlsx","xls"]:
                            output = pd.read_excel(temp_filepath)

                        elif ext == 'txt':
                            with open(temp_filepath, "r") as f:
                                output = f.read()
                        elif ext == "json":
                            with open(temp_filepath, "r", encoding='utf-8') as f:
                                content = f.read()
                                try:
                                    output = json.loads(content)
                                except json.JSONDecodeError:
                                    lines = content.strip().split('\n')
                                    output = [json.loads(line) for line in lines if line]
                        elif ext in ('yaml', 'yml'):
                            with open(temp_filepath, 'r') as f:
                                output = yaml.safe_load(f)
                        elif ext == "pkl":
                            output = joblib.load(temp_filepath)
                        else:
                            output = None

                        self._logger.info(f'Read in {name}')
                        return output

                    except Exception as e:
                        self._logger.error(f"Failed to read file {name}: {e}")
                        raise e
                    finally:
                        if os.path.exists(temp_filepath):
                            os.remove(temp_filepath)
                else:
                    raise ValueError(f'File {name} does not have a valid extension. Valid extensions are: {valid_ext}')

        except Exception as e:
            self._logger.error(f"Failed to download blob {source_blob_name} from bucket {self._bucket_name}: {e}")
            raise e

    def list_blobs(self, prefix=None):
        '''
        Lists all blobs in the bucket with an optional prefix.
        :param prefix: Optional prefix to filter blobs.
        :return: List of blob names.
        '''
        try:
            bucket = self._client.bucket(self._bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            blob_names = [blob.name for blob in blobs]
            self._logger.info(f"Listed {len(blob_names)} blobs in bucket {self._bucket_name} with prefix '{prefix}'.")
            print(f"Listed {len(blob_names)} blobs in bucket {self._bucket_name} with prefix '{prefix}'.")
            return blob_names
        except Exception as e:
            self._logger.error(f"Failed to list blobs in bucket {self._bucket_name}: {e}")
            raise e
