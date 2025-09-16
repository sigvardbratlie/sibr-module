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
    from google.cloud import bigquery
    import pandas_gbq as pbq
    import joblib
    import yaml
except ImportError:
    raise ImportError(
        "BigQuery krever 'google-cloud-bigquery' og 'pandas-gbq'. "
        "Installer den med: pip install 'sibr-module[bigquery]'"
    )

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