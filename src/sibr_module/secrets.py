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
    from google.cloud import secretmanager
    import tomllib
except ImportError:
    raise ImportError(f'SecretsManager requires "google-cloud-secretmanager"'
                      "Run pip install 'sibr-module[secretsmanager]' to install")


class SecretsManager:
    """
    An extended client for Google Secret Manager.

    This class handles both fetching and uploading secrets.
    It can read from local .env and .toml files and upload them to
    Google Secret Manager, handling nested structures in TOML files
    by flattening the keys.
    """

    def __init__(self, project_id: str = None, logger = None):
        """
        Initializes the SecretsManager client.

        It automatically discovers the project_id from the GOOGLE_APPLICATION_CREDENTIALS
        environment variable if not provided.

        Args:
            project_id (str, optional): Your Google Cloud Project ID.
            logger (Logger, optional): A logger instance for logging messages.
        """

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
        self.logger = logger or logging.getLogger("SecretsManager")
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