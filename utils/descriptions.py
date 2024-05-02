import json
import logging
import os
from pathlib import Path
from typing import Dict, NewType, Optional, Union

import pandas as pd
import requests

from utils.checksum import perform_checksum

DESCRIPTION_FILE_NAME = "Cap3D_automated_Objaverse_full.csv"
CAPTIONS_URL = (
    f"https://huggingface.co/datasets/tiange/Cap3D/resolve/main/{DESCRIPTION_FILE_NAME}"
)
REQUESTS_TIMEOUT = 60
CHUNK_SIZE = 8192

PositiveInt = NewType("PositiveInt", int)


def stream_download_file(
    url: str,
    file: Path,
    requests_timeout: Optional[PositiveInt] = 60,
    chunk_size: Optional[PositiveInt] = 8192,
) -> None:
    """
    Streams data from a specified URL and writes it to a local file in "append binary" mode.

    This function downloads a file from the given URL and writes it to the specified local file
    using a streaming approach. The streaming ensures that large files can be handled efficiently
    without consuming excessive memory.

    Parameters
    ----------
    url : str
        The URL from which to download the file.
    file : Path
        The local file path where the downloaded data will be written.
    requests_timeout : PositiveInt, optional
        The timeout for the HTTP request in seconds. Defaults to 60.
    chunk_size : PositiveInt, optional
        The size of data chunks to read at a time during file download. Defaults to 8192.

    Raises
    ------
    requests.exceptions.RequestException
        If there's an issue with the HTTP request, such as a network error, invalid URL, or timeout.
    """
    with requests.get(url, stream=True, timeout=requests_timeout) as response:
        # Raise exception if status code is anything other than 200
        response.raise_for_status()

        # Write to file in "append binary" mode ("ab")
        with file.open("ab") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def get_latest_descriptions(
    description_file_name: Union[str, os.PathLike] = DESCRIPTION_FILE_NAME,
    captions_url: str = CAPTIONS_URL,
    requests_timeout: PositiveInt = REQUESTS_TIMEOUT,
    chunk_size: PositiveInt = CHUNK_SIZE,
    performing_checksum: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """
    Downloads a CSV file containing dataset descriptions from a specified URL, saves it to disk,
    and creates a dictionary mapping dataset UIDs to their corresponding descriptions.

    If the local file does not exist or if it is outdated based on a checksum comparison,
    the function downloads the latest CSV from the provided URL. After downloading, it parses the
    CSV to create a dictionary where keys are dataset unique identifiers (UIDs) and values are
    their descriptions.

    Parameters
    ----------
    description_file_name : Union[str, os.PathLike], optional
        The name or path to the file where the CSV will be saved. Defaults to `DESCRIPTION_FILE_NAME`.
    captions_url : str, optional
        The URL to download the CSV file from. Defaults to `CAPTIONS_URL`.
    requests_timeout : PositiveInt, optional
        The timeout for the HTTP request in seconds. Defaults to 60.
    chunk_size : PositiveInt, optional
        The size of data chunks to read at a time during file download. Defaults to 8192.
    performing_checksum : bool, optional
        Flag to dictat whether checksum validation process is performed.
    logger : Optional[logging.Logger], optional
        An instance of `logging.Logger` for logging events during execution. If not provided,
        a new logger with default settings will be created.

    Returns
    -------
    Dict[str, str]
        A dictionary where the keys are unique dataset identifiers (UIDs) and the values are
        their corresponding descriptions.

    Raises
    ------
    requests.exceptions.RequestException
        If there's an issue with the HTTP request (e.g., network error, invalid URL, timeout, etc.).
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

    logger.info(
        "Get latest descriptions process started for file: '%s'", description_file_name
    )

    DESCRIPTION_FILE_PATH: Path = Path(description_file_name)

    pointer_file_url = captions_url.replace("resolve", "raw").replace(
        "?download=true", ""
    )

    logger.info(
        "Get latest descriptions process started for file: '%s'", description_file_name
    )

    if not DESCRIPTION_FILE_PATH.exists() or (
        DESCRIPTION_FILE_PATH.exists()
        and (
            performing_checksum
            and not perform_checksum(
                file_path=description_file_name, pointer_file_url=pointer_file_url
            )
        )
    ):
        logger.info(
            "Descriptions file either does not exists or newer file found, downloading latest file"
        )
        stream_download_file(
            url=captions_url,
            file=DESCRIPTION_FILE_PATH,
            requests_timeout=requests_timeout,
            chunk_size=chunk_size,
        )

    logger.info(f"Reading {'latest' if performing_checksum else ''} descriptions file")
    # Read the CSV into a DataFrame
    df = pd.read_csv(
        DESCRIPTION_FILE_PATH, header=None, names=["datasetUID", "description"]
    )

    # Drop duplicates based on 'datasetUID', keeping the first occurrence
    df.drop_duplicates(subset="datasetUID", keep="first", inplace=True)

    logger.info("Generating hashmap from latest descriptions file")
    # Create a dictionary with 'datasetUID' as keys and 'description' as values
    descriptions_dict: Dict[str, str] = dict(zip(df["datasetUID"], df["description"]))

    logger.info(
        "Get latest descriptions process successfuly completed for file: '%s'",
        description_file_name,
    )

    return descriptions_dict
