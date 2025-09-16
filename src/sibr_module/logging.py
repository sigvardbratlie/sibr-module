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
    from google.cloud.logging_v2.handlers import CloudLoggingHandler
    from google.cloud import logging as cloud_logging
except ImportError:
    raise ImportError(
        "Logger module requires 'google-cloud-logging_v2-handlers-CloudLoggingHandler' and 'google-cloud-logging-cloud-logging"
        "Run pip install 'sibr-module[logging] to install"
    )

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




