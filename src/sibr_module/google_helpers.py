import random

import pandas as pd
from google.cloud import bigquery
from google.cloud import storage
import uuid
import pandas_gbq as pbq
import numpy as np
from typing import Literal
import traceback
from google.cloud import secretmanager
from google.cloud import logging as cloud_logging
from google.auth.exceptions import DefaultCredentialsError
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.logging_v2.handlers import CloudLoggingHandler
import logging
from pathlib import Path
import os
import joblib
import json
import yaml

try:
    import tomllib
except ImportError:
    import tomli as tomllib


class Logger:
    """A flexible logger for both local and cloud-based logging.

    This class sets up a logger that can write to a local file and/or
    send logs to Google Cloud Logging. It automatically handles log file
    creation and the cloud connection setup.

    Attributes:
        log_level: Property to get or set the logging level (e.g., 'INFO', 'DEBUG').
        enable_cloud_logging (bool): Indicates if Google Cloud Logging is active.
    """
    def __init__(self, log_name : str,
                 root_path : str =None,
                 write_type : Literal['a','w'] = 'a',
                 enable_cloud_logging : bool = False,
                 cloud_log_name : str = None):
        '''
        Initialize a logger.
        :param log_name:
        :param root_path: Optional. Default is None
        :param write_type: A option to either replace or append to logs if the file already exists
        :param enable_cloud_logging: Set to True for cloud logging
        :param cloud_log_name: A specific log name for cloud logging. Variable is set to log_name if not specified.
        '''
        self._logName = log_name
        if write_type not in ["a","w"]:
            raise ValueError("write_type must be either 'a' or 'w'")
        if root_path:
            self._root = root_path
        elif os.getenv('DOCKER'):
            self._root = Path('/app')  # Fortsatt relevant hvis du har Docker-spesifikk logikk
        else:
            self._root = Path.cwd()
        self.enable_cloud_logging = enable_cloud_logging
        self._path = self._create_log_folder() / f'{self._logName}.log'
        self._write_type = write_type
        self._logger = logging.getLogger(log_name)
        self._logger.setLevel(logging.INFO)
        self._path = self._create_log_folder() / f'{self._logName}.log'
        if self._logger.hasHandlers():
            self._logger.handlers.clear()
        self._create_handlers()
        if enable_cloud_logging:
            if cloud_log_name is None:
                cloud_log_name = log_name
            self.cloud_log_name = cloud_log_name
            self._logger.info(f'Cloud Logging is enabled. Initializing Google Cloud Logging for {self._logName} with cloud_log_name {self.cloud_log_name}.')
            self._setup_cloudlogging()
        else:
            self._logger.info(f'Cloud Logging is disabled. Using local logging to {self._path}.')


    def _setup_cloudlogging(self):
        self.client = None
        try:
            self.client = cloud_logging.Client()
            self.client.setup_logging()
            cloud_handler = CloudLoggingHandler(self.client, name=self.cloud_log_name)
            cloud_handler.setLevel(logging.DEBUG)
            self._logger.addHandler(cloud_handler)
            self._logger.info(f'Google Cloud Logging initialized with project: {self.client.project}')
            self._logger.info(f'All loggs successfully initiated. Current dir: {os.getcwd()}.')
        except DefaultCredentialsError as e:
            self.client = None
            self._logger.error(f'Authentication error initializing Google Cloud Logging: {e}', exc_info=True)
            self._logger.warning('Cloud Logging will not be available due to authentication issues.')
        except GoogleAPICallError as e:
            self.client = None
            self._logger.error(f'Google API Call Error during Cloud Logging initialization: {e}', exc_info=True)
            self._logger.warning(
                'Cloud Logging might not be available due to an API error. Check permissions or network.')
        except Exception as e:
            self.client = None
            self._logger.error(f'An unexpected error occurred during Cloud Logging initialization: {e}', exc_info=True)
            self._logger.warning('Cloud Logging will not be available due to an unexpected error.')

    @property
    def log_level(self):
        return self._logger.getEffectiveLevel()
    @log_level.setter
    def log_level(self, level):
        if level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValueError(f'Invalid log level: {level}. Choose between DEBUG, INFO, WARNING, ERROR, CRITICAL')
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)

    def shutdown(self):
        for handler in self._logger.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

    def _create_handlers(self):
        file_handler = logging.FileHandler(self._path, mode=self._write_type)
        console_handler = logging.StreamHandler()

        file_handler.setLevel(logging.DEBUG)  # Log all levels to the file
        console_handler.setLevel(logging.DEBUG)  # Log only warnings and above to the console

        # Create formatters and add them to the handlers
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        # Add handlers to the logger
        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)

    def _create_log_folder(self):
        if self._root:
            root = self._find_root_folder()
            path = root / 'logfiles'
        else:
            path = Path.cwd() / 'logfiles'
        if not path.exists():
            path.mkdir()
        return path

    def _find_root_folder(self):
        if os.getenv('DOCKER'):
            return Path('/app')
        current_path = Path.cwd()
        path = current_path
        while True:
            for file in path.iterdir():
                if '.venv' in file.name:
                    return path
            if path == path.parent:
                break
            path = path.parent
        return current_path


    def debug(self, msg: str):
        self._logger.debug(msg)

    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def critical(self, msg: str):
        self._logger.critical(msg)

    def set_level(self, level):
        if level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValueError(f'Invalid log level: {level}. Choose between DEBUG, INFO, WARNING, ERROR, CRITICAL')
        self._logger.setLevel(level)

class BigQuery:
    """A helper class for interacting with Google BigQuery.

        This class simplifies common BigQuery operations such as uploading
        pandas DataFrames to tables and running SQL queries to fetch data.
        It includes built-in logic for 'append', 'replace', and 'merge' operations.

        Attributes:
            project (str): The Google Cloud project ID associated with the client.
        """
    def __init__(self, project_id : str = None, logger = None, dataset : str = None):
        '''
        Initialize a BigQuery client.
        :param project_id: Your Google Cloud project ID.
        :param logger: Optional, a Logger instance for logging.
        :param dataset: Optional, the name of the BigQuery dataset.
        '''
        if not logger:
            logger = Logger("BigQuery")
        self._logger = logger
        self.dataset = dataset
        if project_id is None:
            cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            with open(cred_path,"r") as f:
                cred = json.load(f)
                project_id = cred.get("project_id")
        self.project = project_id
        try:
            self._bq_client = bigquery.Client(project=self.project)
            self._credentials = self._bq_client._credentials

            self._logger.info(f"BigQuery client initialized with project_id: {self.project}")
        except Exception as e:
            self._logger.error(f"Error initializing BigQuery client: {e}")
            self._logger.error(traceback.format_exc())
            raise ImportError(f"Error initializing BigQuery client: {e}, cwd: {os.getcwd()}")

    def _clean_and_prepare_df(self, df: pd.DataFrame, schema_mapping: dict) -> pd.DataFrame:
        """Renser og klargjør DataFrame i henhold til BigQuery-skjemaet."""
        df_copy = df.copy()
        for column, bq_type in schema_mapping.items():
            if column not in df_copy.columns:
                continue

            # Tving kolonner til riktig type og erstatt feil med NaN/NaT
            if bq_type in ['INTEGER', 'FLOAT']:
                df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
                # For Integer, må vi bruke Pandas' nullable integer type for å beholde NaN
                if bq_type == 'INTEGER':
                    df_copy[column] = df_copy[column].astype('Int64')  # Nullable Integer
            elif bq_type in ['DATETIME', 'TIMESTAMP', 'DATE']:
                df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce', utc=True)
                # Konverter til None der det er NaT (Not a Time), som BigQuery-klienten håndterer
                df_copy[column] = df_copy[column].replace({pd.NaT: None})

        self._logger.info("DataFrame cleaned and prepared for BigQuery upload.")
        return df_copy

    def _get_dtype(self, df: pd.DataFrame, column_name: str):
        """
        Finds the Python data type of the first non-null element in a DataFrame column.

        :param df: The DataFrame.
        :param column_name: The name of the column to check.
        :return: The name of the data type (e.g., 'str', 'int'), or None if the column is empty or all-null.
        """
        # Lag en ny serie uten null-verdier (NaN, None, etc.)
        non_null_series = df[column_name].dropna()

        # Hvis serien er tom etter fjerning av null-verdier, returner None
        if non_null_series.empty:
            self._logger.warning(f"Column '{column_name}' contains only null values or is empty.")
            return None

        # Hent det aller første elementet som ikke er null og returner typenavnet
        first_valid_element = non_null_series.iloc[0]
        return type(first_valid_element).__name__

    def to_bq(self,
              df : pd.DataFrame,
              table_name : str,
              dataset_name : str,
              if_exists: Literal['append', 'replace', 'merge'] = 'append',
              to_str=False,
              merge_on=None,
              autodetect : bool = False,
              dtype_map : dict = None,
              explicit_schema : dict = None):
        '''
        Save a DataFrame to BigQuery.
        :param df:
        :param table_name:
        :param dataset_name:
        :param if_exists: Choose between 'append', 'replace', or 'merge'.
        :param to_str: Optional to convert all columns to string.
        :param merge_on: Required with if_exists = 'merge'.
        :param autodetect: boolean.
        :param dtype_map: Optional. Add a map from datatypes to desired BigQuery types as a dictionary. Examples: dtype_map = {'object': 'STRING','string': 'STRING','int64': 'INTEGER'}
        :param explicit_schema: Optional. Add desired types to specific columns in your dataframe.
        :return:

        The default dtype_map is defined as
            dtype_map = {
                'object': 'STRING',
                'string': 'STRING',
                'category': 'STRING',
                'str': 'STRING',
                'list': ("STRING", "REPEATED"),
                'int64': 'INTEGER',
                'Int64': 'INTEGER',
                'int64[pyarrow]': 'INTEGER',
                'float32' : 'FLOAT',
                'Float32' : 'FLOAT',
                'float64': 'FLOAT',
                'Float64': 'FLOAT',
                'bool': 'BOOLEAN',
                'boolean': 'BOOLEAN',
                'decimal.Decimal': "NUMERIC",
                'Decimal': "NUMERIC",
                'datetime64[ns]': 'DATETIME',
                'datetime': 'DATETIME',
                'datetime64[ns, UTC]': 'TIMESTAMP',
                'Timestamp': 'TIMESTAMP',
                'date32[day][pyarrow]': 'DATE',
                'datetime64[us]': 'DATETIME',
            }
        '''
        dataset_id = f'{self.project}.{dataset_name}'
        table_id = f"{dataset_id}.{table_name}"
        if if_exists not in ['append', 'replace', 'merge']:
            raise TypeError(f"Invalid if_exists value: {if_exists}. Choose between 'append', 'replace', or 'merge'.")
        if dtype_map is not None and not isinstance(dtype_map, dict):
            raise TypeError(f"Invalid dtype_map value: {dtype_map}. Expected a dictionary.")
        if explicit_schema is not None and not isinstance(explicit_schema, dict):
            raise TypeError(f"Invalid explicit_schema value: {explicit_schema}. Expected a dictionary.")

        try:
            self._bq_client.get_table(table_id)
            table_exists = True
        except Exception:
            table_exists = False

        schema = None
        column_to_bq_type = {}
        if not autodetect:
            if dtype_map is None:
                dtype_map = {
                    'object': 'STRING',
                    'string': 'STRING',
                    'category': 'STRING',
                    'str': 'STRING',
                    'list': ("STRING", "REPEATED"),
                    'int' : 'INTEGER',
                    'int64': 'INTEGER',
                    'Int64': 'INTEGER',
                    'int64[pyarrow]': 'INTEGER',
                    'float' : 'FLOAT',
                    'float32': 'FLOAT',
                    'Float32': 'FLOAT',
                    'float64': 'FLOAT',
                    'Float64': 'FLOAT',
                    'bool': 'BOOLEAN',
                    'boolean': 'BOOLEAN',
                    'decimal.Decimal': "NUMERIC",
                    'Decimal': "NUMERIC",
                    'datetime64[ns]': 'DATETIME',
                    'datetime': 'DATETIME',
                    'date' : 'DATE',
                    'datetime64[ns, UTC]': 'TIMESTAMP',
                    'Timestamp': 'TIMESTAMP',
                    'date32[day][pyarrow]': 'DATE',
                    'datetime64[us]': 'DATETIME',
                    'geometry' : "GEOGRAPHY",
                }

            if explicit_schema is None:
                explicit_schema = {}

            schema = []

            for column_name, dtype in df.dtypes.items():
                correct_dtype = self._get_dtype(df = df,
                                                column_name=str(column_name))
                bq_spec = explicit_schema.get(column_name, dtype_map.get(correct_dtype, 'STRING'))
                if correct_dtype not in dtype_map.keys() and correct_dtype is not None:
                    self._logger.warning(f"No mapping from {correct_dtype} to Big Query types for column {column_name}. Current mapping: {dtype_map}")

                if isinstance(bq_spec, tuple):
                    bq_type, bq_mode = bq_spec
                else:
                    bq_type = bq_spec
                    bq_mode = "NULLABLE"

                schema.append(bigquery.SchemaField(str(column_name), bq_type, mode=bq_mode))
                column_to_bq_type[column_name] = bq_type

            df = self._clean_and_prepare_df(df, column_to_bq_type)

        if if_exists in ['append', 'replace']:

            if if_exists == 'append':
                if not table_exists:
                    self._logger.warning(f"Table {table_id} does not exist. Creating a new table.")
                job_config = bigquery.LoadJobConfig(
                    write_disposition="WRITE_APPEND" if table_exists else "WRITE_TRUNCATE",
                    schema=schema,
                    autodetect=autodetect,
                )
            elif if_exists == 'replace':
                job_config = bigquery.LoadJobConfig(
                    write_disposition="WRITE_TRUNCATE",
                    schema=schema,
                    autodetect=autodetect,
                )
            try:
                if to_str:
                    df = df.astype(str)
                job = self._bq_client.load_table_from_dataframe(
                    df, table_id, job_config=job_config
                )
                job.result()
                self._logger.info(f"{len(df)} rader lagret i {table_id}")
            except Exception as e:
                self._logger.error(
                    f"Error saving to BigQuery: {type(e).__name__}: {e}")
                self._logger.error(traceback.format_exc())
        elif if_exists == 'merge':

            if not merge_on or not isinstance(merge_on, list):
                raise ValueError(
                    "merge_on parameter must be provided when if_exists is 'merge' and must be a list of column names.")
            staging_table_id = f"{table_id}_staging_{uuid.uuid4().hex}"
            self._logger.info(f"Starting MERGE. Uploading data to staging table: {staging_table_id}")

            try:

                job_config = bigquery.LoadJobConfig(
                    write_disposition="WRITE_TRUNCATE",
                    schema=schema,
                    autodetect=autodetect,
                )
                if to_str:
                    df = df.astype(str)

                job = self._bq_client.load_table_from_dataframe(df, staging_table_id, job_config=job_config)
                job.result()  # Wait for the job to complete
                self._logger.info(f"Staging table {staging_table_id} created with {len(df)} rows.")

                on_condition = ' AND '.join([f'T.`{key}` = S.`{key}`' for key in merge_on])

                update_cols = [col for col in df.columns if col not in merge_on]
                update_set = ', '.join([f'T.`{col}` = S.`{col}`' for col in update_cols])

                insert_cols = ', '.join([f'`{col}`' for col in df.columns])
                insert_values = ', '.join([f'S.`{col}`' for col in df.columns])

                merge_query = f"""
                                MERGE `{table_id}` AS T
                                USING `{staging_table_id}` AS S
                                ON {on_condition}
                                WHEN MATCHED THEN
                                    UPDATE SET {update_set}
                                WHEN NOT MATCHED THEN
                                    INSERT ({insert_cols})
                                    VALUES ({insert_values})
                                """

                self._logger.info("Executing MERGE statement...")
                self.exe_query(merge_query)
                self._logger.info(f"MERGE operation on {table_id} complete.")

            finally:
                self._logger.info(f"Deleting staging table: {staging_table_id}")
                self._bq_client.delete_table(staging_table_id, not_found_ok=True)
        else:
            raise ValueError(f"Invalid if_exists value: {if_exists}")

    def read_bq(self, query : str , read_type: Literal["bigframes", "bq_client", "pandas_gbq"] = 'bq_client'):
        '''
        Read data from BigQuery.
        :param query: SQL query; for examples "SELECT * FROM dataset_name.table_name LIMIT 1000"
        :param read_type: Choose between 'bigframes', 'bq_client' and 'pandas_gbq'. Default is "bq_client"
        :return:
        '''
        if read_type == 'bq_client':
            df = self._bq_client.query(query).to_arrow().to_pandas()
        elif read_type == 'pandas_gbq':
            df = pbq.read_gbq(query, credentials=self._credentials)
        else:
            raise ValueError(
                f"Invalid read_type: {read_type}. Choose between 'bigframes', 'bq_client' and 'pandas_gbq'")
        #df.replace(['nan', 'None', '', 'null', 'NA', '<NA>', 'NaN', 'NAType'], np.nan, inplace=True)
        self._logger.info(f"{len(df)} rader lest fra BigQuery")
        return df

    def exe_query(self, query : str):
        '''
        Execute a BigQuery query
        :param query:
        :return:
        '''
        job = self._bq_client.query(query)
        job.result()
        self._logger.info(f"Query executed: {query[:100]}... (truncated)")

class SecretsManager:
    """A simple client for fetching secrets from Google Secret Manager.

        This class acts as a thin wrapper around Google's official client
        to make the most common operation—retrieving the latest version
        of a secret—as straightforward as possible.

        Attributes:
            project (str): The Google Cloud project ID associated with the client.
            client: The underlying SecretManagerServiceClient instance.
        """
    def __init__(self, project_id: str = None, logger = None):
        '''
        Initialize a SecretsManager client.
        :param project_id:
        :param logger: Optional, a Logger instance for logging.
        '''
        if project_id is None:
            cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            with open(cred_path, "r") as f:
                cred = json.load(f)
                project_id = cred.get("project_id")
        self.project = project_id
        if logger is None:
            logger = Logger("SecretsManager")
        self.logger = logger
        self.client = secretmanager.SecretManagerServiceClient()
        self.logger.info("SecretsManager initialized.")

    def get_secret(self, secret_id: str):
        '''
        Get a secret from Secrets Manager.
        :param secret_id:
        :return:
        '''
        name = {"name": f'projects/{self.project}/secrets/{secret_id}/versions/latest'}
        try:
            response = self.client.access_secret_version(name)
            if response:
                self.logger.info(f"Read in secret: {secret_id} from project {self.project}")
                return response.payload.data.decode("UTF-8")
        except Exception as e:
            self.logger.error(str(e))

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
            logger = Logger("CStorage")
        self._logger = logger
        self._bucket_name = bucket_name
        if project_id is None:
            cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            with open(cred_path, "r") as f:
                cred = json.load(f)
                project_id = cred.get("project_id")
        self.project = project_id
        try:
            self._client = storage.Client(project=project_id)
            self._logger.info(f"Google Cloud Storage client initialized with bucket: {self._bucket_name}")
        except Exception as e:
            self._logger.error(f"Error initializing Google Cloud Storage client: {e}")
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






