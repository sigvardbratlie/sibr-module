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

if TYPE_CHECKING:
    from google.cloud import bigquery
    import pandas_gbq as pbq
    import joblib
    import yaml
    import tomllib
    from google.cloud import storage
    from google.cloud import secretmanager
    import pandas_gbq as pbq
    from google.cloud.logging_v2.handlers import CloudLoggingHandler
    from google.cloud import logging as cloud_logging


class LoggerV2(logging.Logger):
    """
    A flexible logger that inherits from logging.Logger, supporting both
    local and cloud-based logging with an interchangeable interface.

    Attributes:
        log_level: Property to get or set the logging level (e.g., 'INFO', 'DEBUG').
        cloud_logging_enabled (bool): Indicates if Google Cloud Logging is active.
    """
    def __init__(self,
                 name: str,
                 level: int = logging.INFO,
                 root_path: Optional[str] = None,
                 write_type: Literal['a', 'w'] = 'a',
                 enable_cloud_logging: bool = False,
                 cloud_log_name: Optional[str] = None):
        """
        Initialize the logger.
        :param name: Name of the logger, typically __name__.
        :param level: Initial logging level.
        :param root_path: Optional root path for log files. Defaults to project root or CWD.
        :param write_type: 'a' to append to existing logs, 'w' to overwrite.
        :param enable_cloud_logging: Set to True to enable Google Cloud Logging.
        :param cloud_log_name: Specific name for the cloud log stream. Defaults to `name`.
        """
        try:
            from google.cloud.logging_v2.handlers import CloudLoggingHandler
            from google.cloud import logging as cloud_logging
        except ImportError:
            raise ImportError(
                "Logger module requires 'google-cloud-logging_v2-handlers-CloudLoggingHandler' and 'google-cloud-logging-cloud-logging"
                "Run pip install 'sibr-module[logging] to install"
            )
        super().__init__(name, level)

        if write_type not in ["a", "w"]:
            raise ValueError("write_type must be either 'a' or 'w'")
        self._write_type = write_type

        # Bestem rotsti for logger
        self._root = Path(root_path) if root_path else self._find_project_root()
        self._log_dir = self._create_log_folder()
        self.log_file_path = self._log_dir / f'{self.name}.log'

        self.cloud_logging_enabled = enable_cloud_logging
        self.cloud_log_name = cloud_log_name or self.name # Bruk `name` hvis `cloud_log_name` er None

        # Sørg for at vi ikke legger til handlere flere ganger hvis loggeren allerede finnes
        if not self.hasHandlers():
            self._create_local_handlers()
            if self.cloud_logging_enabled:
                self.info(f'Cloud Logging is enabled. Initializing for {self.name} with cloud_log_name {self.cloud_log_name}.')
                self._setup_cloud_logging()
            else:
                self.info(f'Cloud Logging is disabled. Using local logging to {self.log_file_path}.')

    def _setup_cloud_logging(self):
        try:
            client = cloud_logging.Client()
            handler = CloudLoggingHandler(client, name=self.cloud_log_name)
            self.addHandler(handler)
            self.info(f'Google Cloud Logging initialized with project: {client.project}')
        except DefaultCredentialsError as e:
            self.error(f'Authentication error initializing Google Cloud Logging: {e}', exc_info=True)
            self.warning('Cloud Logging is unavailable due to authentication issues.')
        except GoogleAPICallError as e:
            self.error(f'Google API Call Error during Cloud Logging initialization: {e}', exc_info=True)
            self.warning('Cloud Logging is unavailable due to an API error.')
        except Exception as e:
            self.error(f'Unexpected error during Cloud Logging initialization: {e}', exc_info=True)
            self.warning('Cloud Logging is unavailable due to an unexpected error.')

    def set_global_level(self, level: str | int):
        """Sets the logging level for the logger and all its handlers."""
        log_level_int = logging.getLevelName(level.upper()) if isinstance(level, str) else level
        if not isinstance(log_level_int, int):
            raise ValueError(f'Invalid log level: {level}.')

        self.setLevel(log_level_int)
        for handler in self.handlers:
            handler.setLevel(log_level_int)

    def _create_local_handlers(self):
        # Fil-handler
        file_handler = logging.FileHandler(self.log_file_path, mode=self._write_type)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.addHandler(file_handler)

        # Konsoll-handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.addHandler(console_handler)

    def _create_log_folder(self) -> Path:
        path = self._root / 'logfiles'
        path.mkdir(exist_ok=True) # exist_ok=True er enklere enn å sjekke om den finnes
        return path

    def _find_project_root(self) -> Path:
        if os.getenv('DOCKER'):
            return Path('/app')
        # En enklere og mer vanlig måte er å se etter en kjent fil/mappe som .git eller pyproject.toml
        current_path = Path.cwd()
        while current_path != current_path.parent:
            if (current_path / ".git").exists() or (current_path / ".venv").exists() or (current_path / "pyproject.toml").exists():
                return current_path
            current_path = current_path.parent
        return Path.cwd() # Fallback

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
        try:
            from google.cloud.logging_v2.handlers import CloudLoggingHandler
        except ImportError:
            raise ImportError(
                "Logger module requires 'google-cloud-logging_v2-handlers-CloudLoggingHandler'"
                "Install it with: pip install 'sibr-module[logging]"
            )
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
        try:
            from google.cloud import bigquery
            import pandas_gbq as pbq
            import joblib
            import yaml
        except ImportError:
            raise ImportError(
                "BigQuery krever 'google-cloud-bigquery' og 'pandas-gbq'. "
                "Installer den med: pip install 'sibr-module[bigquery]'"
            )
        if not logger:
            logger = logging.getLogger("BigQuery")
        self.logger = logger
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

            self.logger.info(f"BigQuery client initialized with project_id: {self.project}")
        except Exception as e:
            self.logger.error(f"Error initializing BigQuery client: {e}")
            self.logger.error(traceback.format_exc())
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

        self.logger.info("DataFrame cleaned and prepared for BigQuery upload.")
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
            self.logger.warning(f"Column '{column_name}' contains only null values or is empty.")
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
                    "ndarray" : ("STRING", "REPEATED"),
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
                    "Polygon" : "GEOGRAPHY",
                    'Timedelta' : "INTERVAL",
                    "timedelta64[ns]" : "INTERVAL"
                }

            if explicit_schema is None:
                explicit_schema = {}

            schema = []

            for column_name, dtype in df.dtypes.items():
                correct_dtype = self._get_dtype(df = df,
                                                column_name=str(column_name))
                bq_spec = explicit_schema.get(column_name, dtype_map.get(correct_dtype, 'STRING'))
                if correct_dtype not in dtype_map.keys() and correct_dtype is not None:
                    self.logger.warning(f"No mapping from {correct_dtype} to Big Query types for column {column_name}. Current mapping: {dtype_map}")

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
                    self.logger.warning(f"Table {table_id} does not exist. Creating a new table.")
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
                self.logger.info(f"{len(df)} rader lagret i {table_id}")
            except Exception as e:
                self.logger.error(
                    f"Error saving to BigQuery: {type(e).__name__}: {e}")
                self.logger.error(traceback.format_exc())
        elif if_exists == 'merge':

            if not merge_on or not isinstance(merge_on, list):
                raise ValueError(
                    "merge_on parameter must be provided when if_exists is 'merge' and must be a list of column names.")

            duplicates = (df.duplicated(subset=merge_on).sum())
            if duplicates or (duplicates)>0:
                self.logger.warning(f'There are {(duplicates)} duplicates in the dataframe based on the merge_on columns {merge_on}. They will be removed before merging starts')
                df = df.drop_duplicates(subset=merge_on)


            staging_table_id = f"{table_id}_staging_{uuid.uuid4().hex}"
            self.logger.info(f"Starting MERGE. Uploading data to staging table: {staging_table_id}")

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
                self.logger.info(f"Staging table {staging_table_id} created with {len(df)} rows.")

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

                self.logger.info("Executing MERGE statement...")
                self.exe_query(merge_query)
                self.logger.info(f"MERGE operation on {table_id} complete.")

            finally:
                self.logger.info(f"Deleting staging table: {staging_table_id}")
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
        self.logger.info(f"{len(df)} rader lest fra BigQuery")
        return df

    def exe_query(self, query : str):
        '''
        Execute a BigQuery query
        :param query:
        :return:
        '''
        job = self._bq_client.query(query)
        job.result()
        self.logger.info(f"Query executed: {query[:100]}... (truncated)")

class SecretsManager:
    """
    An extended client for Google Secret Manager.

    This class handles both fetching and uploading secrets.
    It can read from local .env and .toml files and upload them to
    Google Secret Manager, handling nested structures in TOML files
    by flattening the keys.
    """

    def __init__(self, project_id: str = None, logger: Optional[Logger] = None):
        """
        Initializes the SecretsManager client.

        It automatically discovers the project_id from the GOOGLE_APPLICATION_CREDENTIALS
        environment variable if not provided.

        Args:
            project_id (str, optional): Your Google Cloud Project ID.
            logger (Logger, optional): A logger instance for logging messages.
        """
        try:
            from google.cloud import secretmanager
            import tomllib
        except ImportError:
            raise ImportError(f'SecretsManager requires "google-cloud-secretmanager"'
                              "Run pip install 'sibr-module[secretsmanager]' to install")
        if project_id is None:
            cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if cred_path and os.path.exists(cred_path):
                with open(cred_path, "r") as f:
                    cred = json.load(f)
                    project_id = cred.get("project_id")
            if not project_id:
                raise ValueError(
                    "Could not determine project_id. Please provide it or set GOOGLE_APPLICATION_CREDENTIALS.")

        self.project_id = project_id
        self.logger = logger or Logger("SecretsManager")
        self.parent = f"projects/{self.project_id}"
        try:
            self.client = secretmanager.SecretManagerServiceClient()
            self.logger.info(f"SecretsManager initialized for project '{self.project_id}'.")
        except Exception as e:
            self.logger.error(f"Could not initialize SecretManagerServiceClient: {e}")
            raise

    def _create_or_update_secret(self, secret_id: str, secret_value: str):
        """
        Creates a secret if it doesn't exist, then adds a new version.
        This is an internal helper method.

        Args:
            secret_id (str): The ID for the secret.
            secret_value (str): The value of the secret to add.
        """
        secret_path = self.client.secret_path(self.project_id, secret_id)
        try:
            self.client.get_secret(request={"name": secret_path})
            self.logger.info(f"Secret '{secret_id}' already exists. Adding a new version.")
        except NotFound:
            self.logger.info(f"Creating new secret: '{secret_id}'")
            try:
                self.client.create_secret(
                    request={
                        "parent": self.parent,
                        "secret_id": secret_id,
                        "secret": {"replication": {"automatic": {}}},
                    }
                )
            except AlreadyExists:
                self.logger.warning(f"Secret '{secret_id}' was created by another process in the meantime. Continuing.")

        # Add the new secret version
        payload = secret_value.encode("UTF-8")
        self.client.add_secret_version(
            request={"parent": secret_path, "payload": {"data": payload}}
        )
        self.logger.info(f" -> Successfully added new version for '{secret_id}'.")

    def upload_secrets_from_env(self, filepath: str = ".env"):
        """
        Uploads all key-value pairs from a .env file to Secret Manager.

        Args:
            filepath (str, optional): The path to the .env file. Defaults to ".env".
        """
        if not os.path.exists(filepath):
            self.logger.error(f"File not found: {filepath}")
            return

        self.logger.info(f"--- Starting upload from {filepath} ---")
        secrets_to_upload = dotenv_values(filepath)
        if not secrets_to_upload:
            self.logger.warning(f"No secrets found in {filepath}.")
            return

        for secret_id, secret_value in secrets_to_upload.items():
            if secret_id=="GOOGLE_APPLICATION_CREDENTIALS":
                continue
            if secret_value:  # Do not upload empty values
                self._create_or_update_secret(secret_id.upper(), secret_value)
        self.logger.info(f"--- Finished upload from {filepath} ---")

    def upload_secrets_from_toml(self, filepath: str):
        """
        Uploads secrets from a .toml file, handling nested structures.
        Nested keys are flattened using a double underscore (e.g., [auth.google] -> AUTH__GOOGLE).

        Args:
            filepath (str): The path to the .toml file (e.g., ".streamlit/secrets.toml").
        """
        if not os.path.exists(filepath):
            self.logger.error(f"File not found: {filepath}")
            return

        self.logger.info(f"--- Starting upload from {filepath} ---")
        with open(filepath, "rb") as f:
            data = tomllib.load(f)

        # Use the recursive helper to flatten and upload
        self._upload_nested_dict(data)
        self.logger.info(f"--- Finished upload from {filepath} ---")

    def _upload_nested_dict(self, data: dict[str, Any], prefix: str = ""):
        """
        A recursive helper function to handle nested dictionaries from the TOML file.
        """
        for key, value in data.items():
            # Create the new flattened key, e.g., "AUTH" -> "AUTH__GOOGLE"
            new_key = f"{prefix}{key.upper()}"
            if isinstance(value, dict):
                # If the value is another dictionary, recurse
                self._upload_nested_dict(value, f"{new_key}__")
            else:
                # If it's a final value, upload it
                self._create_or_update_secret(new_key, str(value))

    def get_secret(self, secret_id: str, version: str = "latest", default: Optional[str] = None) -> Optional[str]:
        """
        Retrieves the value of a specific secret.

        Args:
            secret_id (str): The ID of the secret (e.g., "OPENAI_API_KEY").
            version (str, optional): The version number or "latest". Defaults to "latest".
            default (str, optional): The default value to return if the secret is not found.
                                     If None, the error is logged and None is returned.

        Returns:
            Optional[str]: The secret's value, or the default value/None if not found.
        """
        name = f'projects/{self.project_id}/secrets/{secret_id}/versions/{version}'
        try:
            response = self.client.access_secret_version(name=name)
            value = response.payload.data.decode("UTF-8")
            self.logger.info(f"Successfully accessed secret '{secret_id}' (version: {version}).")
            return value
        except NotFound:
            self.logger.warning(f"Secret '{secret_id}' (version: {version}) not found. Returning default value.")
            return default
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while accessing secret '{secret_id}': {e}")
            return default

    def delete_secrets(self, secret_ids: list[str]):
        for secret_id in secret_ids:
            try:
                self.client.delete_secret(request = {"name": self.client.secret_path(self.project_id, secret_id)})
                self.logger.info(f"Successfully deleted secret '{secret_id}'.")
            except NotFound as e:
                self.logger.error(f"Could not delete secret '{secret_id}': {e}")
            except Exception as e:
                self.logger.error(f'Error when deleting secret: {e}')

    def get_secrets(self, secret_ids: List[str]) -> Dict[str, Optional[str]]:
        """
        Retrieves multiple secrets at once and returns them as a dictionary.

        Args:
            secret_ids (List[str]): A list of secret IDs to retrieve.

        Returns:
            Dict[str, Optional[str]]: A dictionary mapping secret IDs to their values.
        """
        self.logger.info(f"Attempting to fetch {len(secret_ids)} secrets.")
        secrets = {secret_id: self.get_secret(secret_id) for secret_id in secret_ids}
        return secrets

    def list_secrets(self) -> List[str]:
        """
        Lists all secret IDs available in the project.

        Returns:
            List[str]: A list of all secret IDs.
        """
        secret_ids = []
        try:
            for secret in self.client.list_secrets(parent=self.parent):
                # secret.name is in the format 'projects/PROJECT_ID/secrets/SECRET_ID'
                secret_id = secret.name.split("/")[-1]
                secret_ids.append(secret_id)
            self.logger.info(f"Found {len(secret_ids)} secrets in project '{self.project_id}'.")
            return secret_ids
        except Exception as e:
            self.logger.error(f"Failed to list secrets: {e}")
            return []

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
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "CStorage krever 'google-cloud-storage'. "
                "Install it with: pip install 'sibr-module[storage]'"
            )
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




# class GoogleAPICallError(Exception): pass
# class DefaultCredentialsError(Exception): pass
# class CloudLoggingHandler:
#     def __init__(self, client, name): pass
#     def setLevel(self, level): pass
# class cloud_logging:
#     @staticmethod
#     def Client():
#         # Simuler en feil for testing
#         # raise DefaultCredentialsError("Mock Credentials Error")
#         class MockClient:
#             project = "mock-project"
#             def setup_logging(self): pass
#         return MockClient()




