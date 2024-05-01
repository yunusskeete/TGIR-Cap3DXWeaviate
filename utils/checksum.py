"""
Utility module for performing checksum operations and verifying (downloaded) files using SHA-256.

This module provides several functions for calculating SHA-256 hashes, extracting them from pointer files, and performing checksum verification against a known expected value.

Example:
    To perform a checksum verification against a known pointer file:

    ```bash
    python checksum.py /path/to/file http://example.com/pointer_file_url
    ```

Functions:
- sha256_hash(data: bytes) -> str:
    Computes the SHA-256 hash of the given data and returns it as a hexadecimal string.

- extract_sha256_from_pointer_file(bytes_data: bytes) -> Optional[str]:
    Extracts the SHA-256 hash from a pointer file given as a byte stream.

- calculate_file_hash(file_path: str, logger: logging.Logger) -> str:
    Calculates the SHA-256 hash of a file at the specified path.

- perform_checksum(
    file_path: str,
    pointer_file_url: str,
    logger: Optional[logging.Logger] = None,
) -> bool:
    Performs a checksum verification of a file against a pointer file. Returns True if the checksum matches, False otherwise.

- main():
    Command-line interface for performing checksum operations. Parses command-line arguments and initiates checksum verification.
"""

import argparse
import hashlib
import logging
import sys
import traceback
from typing import List, Optional

import requests


def sha256_hash(data: bytes) -> str:
    """
    Compute the SHA-256 hash of the given data.

    Args:
        data (bytes): The input data to be hashed.

    Returns:
        str: The hexadecimal representation of the SHA-256 hash.
    """
    sha256: hashlib.sha256 = hashlib.sha256()
    sha256.update(data)

    return sha256.hexdigest()


def extract_sha256_from_pointer_file(bytes_data: bytes) -> str or None:
    """
    Extract the SHA256 hash from a bytes string.

    Args:
        bytes_data (bytes): The bytes string containing the data.

    Returns:
        str or None: The extracted SHA256 hash if found, otherwise None.
    """
    # Decode bytes to string
    data_str: str = bytes_data.decode("utf-8")

    # Split the string into lines
    lines: List[str] = data_str.split("\n")

    # Find the line containing the SHA256 hash
    sha256_line: str or None = next(
        (line for line in lines if line.lower().startswith("oid sha256:")), None
    )

    # Extract the SHA256 hash value
    if sha256_line:
        return sha256_line.split("sha256:")[1]


def calculate_file_hash(file_path: str, logger: logging.Logger) -> str:
    """
    Calculate the SHA256 hash of a file.

    Args:
        file_path (str): The path to the file.
        logger (logging.Logger): Logger instance for logging.

    Returns:
        str: The hexadecimal representation of the SHA256 hash.
    """
    logger.debug(f"Reading file bytes for checksum (file_path: {file_path})")

    # Read the file's bytes
    with open(file_path, "rb") as file:
        file_bytes: bytes = file.read()

    logger.debug("Calculating file hash (checksum) from file bytes")

    # Calculate the SHA256 hash of the file's bytes
    file_hash: str = sha256_hash(file_bytes)

    logger.debug("Successfully calculated file hash (checksum) from file bytes")

    return file_hash


def perform_checksum(
    file_path: str, pointer_file_url: str, logger: Optional[logging.Logger] = None
) -> bool:
    """
    Perform checksum verification of a file against a pointer file.

    Args:
        file_path (str): The path to the file.
        pointer_file_url (str): The URL of the pointer file.
        logger (Optional[logging.Logger]): Logger instance for logging. If not provided, a new one will be created.

    Returns:
        bool: True if the file's checksum matches the expected checksum from the pointer file, False otherwise.
    """
    if not logger or not logger.hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
            ],
        )

        logger = logging.getLogger()  # Get the root logger

    logger.info("Calculate file hash process started for file: '%s'", file_path)
    file_hash: str = calculate_file_hash(file_path=file_path, logger=logger)
    logger.info(
        "Calculate file hash process successfully completed for file: '%s' (file hash: '%s')",
        file_path,
        file_hash,
    )

    logger.info("Requesting pointer file from url: '%s'", pointer_file_url)
    response: requests.Response = requests.get(pointer_file_url, timeout=10)
    response.raise_for_status()
    logger.info("Received pointer file from url: '%s'", pointer_file_url)

    pointer_file: bytes = response.content

    logger.info("Extracting expected file hash from pointer file process started")
    expected_hash: str or None = extract_sha256_from_pointer_file(
        bytes_data=pointer_file
    )
    logger.info(
        "Extracting expected file hash from pointer file process successfully completed (pointer file hash: '%s')",
        expected_hash,
    )

    return file_hash == expected_hash


def main():
    try:
        parser = argparse.ArgumentParser(description="Perform checksum.")
        parser.add_argument("file_path", type=str, help="Path to downloaded file")
        parser.add_argument(
            "pointer_file_url", type=str, help="URL of pointer file for downloaded file"
        )

        args: argparse.Namespace = parser.parse_args()
        print(f"args: {args}")

        file_path: str = args.file_path
        pointer_file_url: str = args.pointer_file_url

        pointer_file_url: str = pointer_file_url.replace("resolve", "raw").split("?")[0]

        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
            ],
        )

        logger = logging.getLogger()  # Get the root logger
        print(f"handlers: {logger.handlers}")
        assert logger.handlers

        checksum_succeeded: bool = perform_checksum(
            file_path=file_path,
            pointer_file_url=pointer_file_url,
            logger=logger,
        )

        logger.info("checksum succeeded: %s", checksum_succeeded)

    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received. Exiting gracefully.")

    except Exception as e:
        # Log the exception traceback
        traceback_str = traceback.format_exc()
        logger.error("An unexpected error occurred: %s", e)
        logger.error("Traceback: %s", traceback_str)

        # Gracefully exit with an appropriate error message
        sys.exit("Exiting due to an unexpected error.")


if __name__ == "__main__":
    main()
