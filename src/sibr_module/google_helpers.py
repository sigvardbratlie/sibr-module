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
    def __init__(self, log_name, root_path=None, write_type='a', enable_cloud_logging=False, cloud_log_name = None):
        '''

        :param log_name:
        :param write_type: can take inn "w" or "a" for write or append
        '''
        self._logName = log_name
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
            self.setup_cloudlogging()
        else:
            self._logger.info(f'Cloud Logging is disabled. Using local logging to {self._path}.')


    def setup_cloudlogging(self):
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
    def __init__(self, project_id : str, logger = None):
        if not logger:
            logger = Logger("BigQuery")
        self._logger = logger
        try:
            self.project = project_id
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

    def _get_dtype(self,df: pd.DataFrame, kolonne_navn: str):
        """
        Finner den faktiske Python-datatypen til det første ikke-null elementet i en DataFrame-kolonne.

        Returnerer:
            type: Datatypen til elementet (f.eks. <class 'str'>, <class 'decimal.Decimal'>),
                  eller None hvis kolonnen er tom eller bare inneholder null-verdier.
        """
        # Lag en ny serie uten null-verdier (NaN, None)
        non_null_series = df[kolonne_navn].dropna()

        # Hvis serien er tom etter å ha fjernet null-verdier, har vi ingen type å sjekke
        if non_null_series.empty:
            return None

        # Hent det første elementet og returner typen
        first_element = non_null_series.iloc[0]
        return type(first_element).__name__

    def to_bq(self, df, table_name, dataset_name, if_exists: Literal['append', 'replace', 'merge'] = 'append',
              to_str=False, merge_on=None):
        dataset_id = f'{self.project}.{dataset_name}'
        table_id = f"{dataset_id}.{table_name}"
        if if_exists not in ['append', 'replace', 'merge']:
            raise ValueError(f"Invalid if_exists value: {if_exists}. Choose between 'append', 'replace', or 'merge'.")
        try:
            self._bq_client.get_table(table_id)
            table_exists = True
        except Exception:
            table_exists = False

        type_mapping = {
            'object': 'STRING',
            'string': 'STRING',
            'int64': 'INTEGER',
            'Int64': 'INTEGER',
            'int64[pyarrow]': 'INTEGER',
            'float64': 'FLOAT',
            'Float64': 'FLOAT',
            'bool': 'BOOLEAN',
            'decimal.Decimal' : "NUMERIC",
            'Decimal' : "NUMERIC",
            'boolean': 'BOOLEAN',
            'datetime64[ns]': 'DATETIME',
            'datetime64[ns, UTC]': 'DATETIME',
            'date32[day][pyarrow]': 'DATE',
            'datetime64[us]': 'DATETIME',
            'category': 'STRING',
            'str' : 'STRING',
            'list' : ("STRING", "REPEATED"),
            'datetime' : 'DATETIME'
        }

        explicit_schema = {
        }

        schema = []
        column_to_bq_type = {}
        for column_name, dtype in df.dtypes.items():
            correct_dtype = self._get_dtype(df, column_name)
            bq_spec = explicit_schema.get(column_name, type_mapping.get(correct_dtype, 'STRING'))

            if isinstance(bq_spec, tuple):
                # Pakk ut type og modus fra tuppelet
                # Dette er nøkkelen! Vi sender type og modus som separate argumenter.
                bq_type, bq_mode = bq_spec
            else:
                # Hvis det er en vanlig kolonne, er typen bare strengen og modusen er standard
                bq_type = bq_spec
                bq_mode = "NULLABLE"  # Standardmodus for vanlige, ikke-påkrevde felter

            # Legg til feltet i skjemaet med riktig type OG riktig modus
            schema.append(bigquery.SchemaField(column_name, bq_type, mode=bq_mode))

            # For vaskefunksjonen trenger vi kun selve datatypen (f.eks. "STRING")
            column_to_bq_type[column_name] = bq_type

        # --- STEG 1: VASK DATAFRAME BASERT PÅ SKJEMA ---
        df = self._clean_and_prepare_df(df, column_to_bq_type)

        if if_exists in ['append', 'replace']:

            if if_exists == 'append':
                if not table_exists:
                    self._logger.warning(f"Table {table_id} does not exist. Creating a new table.")
                job_config = bigquery.LoadJobConfig(
                    write_disposition="WRITE_APPEND" if table_exists else "WRITE_TRUNCATE",
                    schema=schema,  # Bruker eksplisitt schema
                    # autodetect=True,
                )
            if if_exists == 'replace':
                job_config = bigquery.LoadJobConfig(
                    write_disposition="WRITE_TRUNCATE",
                    schema=schema,  # Bruker eksplisitt schema
                    # autodetect=True,
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
                    f"Error saving to BigQuery: {type(e).__name__}: {e} \n for dataframe {df.head()} with columns {df.columns}")
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
                    # autodetect=True,
                )

                if to_str:
                    df = df.astype(str)

                job = self._bq_client.load_table_from_dataframe(df, staging_table_id, job_config=job_config)
                job.result()  # Wait for the job to complete
                self._logger.info(f"Staging table {staging_table_id} created with {len(df)} rows.")

                # Dynamisk bygging av `ON`-betingelsen basert på merge_keys
                on_condition = ' AND '.join([f'T.`{key}` = S.`{key}`' for key in merge_on])

                # Dynamisk bygging av `UPDATE SET`-delen (oppdater alle kolonner unntatt nøklene)
                update_cols = [col for col in df.columns if col not in merge_on]
                update_set = ', '.join([f'T.`{col}` = S.`{col}`' for col in update_cols])

                # Dynamisk bygging av `INSERT`-delen
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
                # self._logger.debug(f'Merge query: {merge_query[:1000]}... (truncated)')

                self._logger.info("Executing MERGE statement...")
                self.exe_query(merge_query)
                self._logger.info(f"MERGE operation on {table_id} complete.")

            finally:
                # --- STEG 3: Slett den midlertidige tabellen ---
                self._logger.info(f"Deleting staging table: {staging_table_id}")
                self._bq_client.delete_table(staging_table_id, not_found_ok=True)
        else:
            raise ValueError(f"Invalid if_exists value: {if_exists}")

    def read_bq(self, query, read_type: Literal["bigframes", "bq_client", "pandas_gbq"] = 'bq_client'):
        '''
        Leser en BigQuery-spørring og returnerer en DataFrame.
        :param query:
        :param read_type: choose between 'bigframes', 'bq_client' and 'pandas_gbq'
        :return:
        '''
        if read_type == 'bq_client':
            df = self._bq_client.query(query).to_arrow().to_pandas()
        elif read_type == 'pandas_gbq':
            df = pbq.read_gbq(query, credentials=self._credentials)
        else:
            raise ValueError(
                f"Invalid read_type: {read_type}. Choose between 'bigframes', 'bq_client' and 'pandas_gbq'")
        df.replace(['nan', 'None', '', 'null', 'NA', '<NA>', 'NaN', 'NAType'], np.nan, inplace=True)
        self._logger.info(f"{len(df)} rader lest fra BigQuery")
        return df

    def exe_query(self, query):
        '''
        Execute a BigQuery query
        :param query:
        :return:
        '''
        job = self._bq_client.query(query)
        job.result()
        self._logger.info(f"Query executed: {query[:100]}... (truncated)")

class SecretsManager:
    def __init__(self, project_id: str, logger = None):
        self.project = project_id
        if logger is None:
            logger = Logger("SecretsManager")
        self.logger = logger
        self.client = secretmanager.SecretManagerServiceClient()
        self.logger.info("SecretsManager initialized.")

    def get_secret(self, secret_id: str):
        name = {"name": f'projects/{self.project}/secrets/{secret_id}/versions/latest'}
        try:
            response = self.client.access_secret_version(name)
            if response:
                self.logger.info(f"Read in secret: {secret_id} from project {self.project}")
                return response.payload.data.decode("UTF-8")
        except Exception as e:
            self.logger.error(str(e))

class CStorage:
    '''
    A class for uploading files to Google Cloud Storage.
    :param bucket_name: The name of the Google Cloud Storage bucket.
    :param logger: An instance of Logger for logging.
    :param CREDENTIALS_PATH: Optional path to a service account key file. If not provided, it will use the default credentials.
    '''

    def __init__(self,project_id,bucket_name, logger = None):
        if logger is None:
            logger = Logger("CStorage")
        self._logger = logger
        self._bucket_name = bucket_name
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
        :return:
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
                            with open(temp_filepath, "r") as f:
                                output = json.load(f)
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






