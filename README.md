

# SIBR Module

A collection of helper modules for interacting with Google Cloud Platform services, designed to simplify common workflows. This package provides easy-to-use classes for BigQuery, Google Cloud Storage, Secret Manager, and Cloud Logging.


## Features

* **BigQuery**: Easily upload DataFrames to BigQuery tables with automatic schema detection and support for append, replace, and merge operations.
* **CStorage**: Upload and download files to and from Google Cloud Storage buckets.
* **SecretsManager**: Securely access secrets from Google Secret Manager.
* **Logger**: A flexible logger that supports both local file logging and integration with Google Cloud Logging.


## Installation

You can install the package from PyPI:

```bash
pip install sibr-module
```


## Quickstart

Ensure you are authenticated with Google Cloud. You can do this by running:

gcloud auth application-default login \


If you are using this locally, make sure to set your credentials as environment variables using: GOOGLE_APPLICATION_CREDENTIALS=path-to-your-credentials.

```python
import pandas as pd 
from sibr_module import BigQuery, Logger 
from dotenv import load_dotenv
load_dotenv()
 ```
### 1. Set up a logger 
This will log to a local file and, if enabled, to Google Cloud Logging \
```python

logger = Logger(log_name="my_app_logger", enable_cloud_logging=True) 
logger.info("Application starting up.")
```

### 2. Prepare your data
```python
data = {'name': ['Alice', 'Bob'], 'score': [85, 92]} 
my_dataframe = pd.DataFrame(data) 
 ```
### 3. Use the BigQuery helper 

```python

try: 
    # Initialize the client with your Google Cloud Project ID
    bq_client = BigQuery(project_id="your-gcp-project-id", logger=logger)

    # Upload the DataFrame to a BigQuery table \
    bq_client.to_bq(
        df=my_dataframe,
        dataset_name="my_dataset", 
        table_name="my_table", 
        if_exists="append"  # Options: 'append', 'replace', or 'merge' 
    ) 
 
    logger.info("Successfully uploaded data to BigQuery.") 
 
except Exception as e: 
    logger.error(f"An error occurred: {e}") 

```
### 3. Use the CStorage helper
```python
try:
    # Create a dummy file to upload
    with open("my_local_file.txt", "w") as f:
        f.write("This is a test file.")

    # Initialize the client
    storage_client = CStorage(project_id="your-gcp-project-id", bucket_name="your-bucket-name", logger=logger)

    # Upload the file
    storage_client.upload(
        local_file_path="my_local_file.txt",
        destination_blob_name="my_remote_folder/my_remote_file.txt"
    )
    logger.info("Successfully uploaded file to GCS.")

    # Download the file
    storage_client.download(
        source_blob_name="my_remote_folder/my_remote_file.txt",
        destination_file_path="downloaded_file.txt"
    )
    logger.info("Successfully downloaded file from GCS.")

except Exception as e:
    logger.error(f"An error occurred with Cloud Storage: {e}")
```


## Usage Details

### BigQuery

The BigQuery class handles interactions with Google BigQuery.



* to_bq(df, dataset_name, table_name, if_exists='append', merge_on=None): Uploads a pandas DataFrame.
    * if_exists='append': Adds data to an existing table.
    * if_exists='replace': Deletes the existing table and creates a new one.
    * if_exists='merge': Updates existing rows and inserts new ones. Requires merge_on to be set with a list of key columns.
* read_bq(query): Executes a query and returns the result as a pandas DataFrame.
* dtype_map: Optional argument if the user wishes to specify a certain mapping of dtypes to BigQuery types.


### CStorage

The CStorage class simplifies file operations with Google Cloud Storage.



* upload(local_file_path, destination_blob_name): Uploads a local file to the bucket.
* download(source_blob_name, destination_file_path): Downloads a file from the bucket.


### SecretsManager

Access your secrets easily.



* get_secret(secret_id): Retrieves the latest version of a secret by its ID.


### Logger

A flexible logger for both local and cloud-based logging.

* __init__(log_name, root_path=None, write_type='a', enable_cloud_logging=False, cloud_log_name=None): Initializes the logger.
* log_level: Property to get or set the logging level (e.g., 'INFO', 'DEBUG').


## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.


## License

This project is licensed under the MIT License. See the LICENSE file for details.