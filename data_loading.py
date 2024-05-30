import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import weaviate
import weaviate.classes as wvc
import weaviate.classes.config as wc
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from weaviate.classes.data import DataObject
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import Filter
from weaviate.collections import Collection
from weaviate.collections.classes.batch import BatchObjectReturn, ErrorObject
from weaviate.connect import ConnectionParams
from weaviate.util import generate_uuid5

from utils.descriptions import get_latest_descriptions
from utils.weaviate import create_collection

HTTP_HOST = "localhost"
HTTP_PORT = 8080
HTTP_SECURE = False

GRPC_HOST = "localhost"
GRPC_PORT = 50051
GRPC_SECURE = False

INTIALISATION_TIMEOUT_S = 2
QUERY_TIMEOUT_S = 45
INSERT_TIMEOUT_S = 120

COLLECTION_NAME = "Cap3DMM"
DATA_UPLOAD_COLLECTION_NAME = "UploadCap3DMM"

BATCH_SIZE = 19
BUFFER_SIZE = 100
PATH_TO_EXAMPLE_OBJECTS = "/home/yunusskeete/Documents/data/3D/Cap3D/local-split/unzips/compressed_imgs_perobj_00.zip/Cap3D_Objaverse_renderimgs"
IMAGE_FILE_EXTENSION = ".png"
IMAGE_FILE_DELIMETER = "_"
IMAGE_ENCODING = "utf-8"
DELETE_ON_UPLOAD = False

PERFORMING_CHECKSUM = False

MODEL_NAME = "clip-ViT-B-32"

path_to_example_objects = Path(PATH_TO_EXAMPLE_OBJECTS)

# pil_logger = logging.getLogger("PIL")
# if pil_logger.hasHandlers():
#     pil_logger.setLevel(logging.INFO)
logging.disable(logging.DEBUG)

descriptions_dict = get_latest_descriptions(
    performing_checksum=PERFORMING_CHECKSUM
)  # https://huggingface.co/datasets/tiange/Cap3D/resolve/48903d63859fe3d3f17942bf6d5383eb05dd1775/Cap3D_automated_Objaverse_full.csv?download=true
print(len(descriptions_dict))


with weaviate.WeaviateClient(
    connection_params=ConnectionParams.from_params(
        http_host=HTTP_HOST,
        http_port=HTTP_PORT,
        http_secure=HTTP_SECURE,
        grpc_host=GRPC_HOST,
        grpc_port=GRPC_PORT,
        grpc_secure=GRPC_SECURE,
    ),
    additional_config=AdditionalConfig(
        timeout=Timeout(
            init=INTIALISATION_TIMEOUT_S, query=QUERY_TIMEOUT_S, insert=INSERT_TIMEOUT_S
        ),  # Values in seconds
    ),
) as client:
    assert client.is_live(), "Weaviate client is not live"
    print("Client connection established")

    try:
        client.collections.delete(COLLECTION_NAME)
        client.collections.delete(DATA_UPLOAD_COLLECTION_NAME)

        cap3d: Collection = create_collection(
            client=client,
            collection_name=COLLECTION_NAME,
            configure_upload_collection=False,
        )
        assert cap3d.aggregate.over_all(total_count=True).total_count == 0

        # Load CLIP model
        model = SentenceTransformer(MODEL_NAME)

        objects_buffer: List[DataObject] = []
        failed_objects: List[List[int, str]] = []

        for object_idx, object_folder in tqdm(
            enumerate(path_to_example_objects.iterdir())
        ):
            object_description: str = descriptions_dict.get(
                object_folder.name, ""
            )  # TODO: Add log if object could not be found, add to tracking list and DO NOT UPLOAD

            object_image_files: List[Path] = [
                file
                for file in object_folder.iterdir()
                if file.suffix == IMAGE_FILE_EXTENSION
                and IMAGE_FILE_DELIMETER not in file.name
            ]

            images: List[Image] = [
                Image.open(object_image) for object_image in object_image_files
            ]

            # Average embeddings from each angle before inserting into database
            try:
                average_embedding: List[float] = np.mean(
                    model.encode(images, show_progress_bar=False),
                    axis=0,
                ).tolist()
            except Exception as e:
                print(f"Numpy exception: {e}")

            # Build the object payload
            object_uuid: str = generate_uuid5(object_folder.name)

            obj: Dict[str, str] = {
                "description": object_description,
                "datasetUID": object_uuid,
            }

            try:
                objects_buffer.append(
                    DataObject(
                        properties=obj,
                        vector=average_embedding,
                        uuid=object_uuid,
                    )
                )
            except Exception as e:
                print(e)
                failed_object_identifier: List[int, str] = [
                    object_idx,
                    object_folder.name,
                ]
                failed_objects.append(failed_object_identifier)
                print(failed_object_identifier)
                print(f"Error appending DataObject to buffer: {e}")

            # Check if the buffer is full
            if len(objects_buffer) >= BUFFER_SIZE:
                try:
                    # Insert into the database
                    batch_objects_return: BatchObjectReturn = cap3d.data.insert_many(
                        objects_buffer
                    )  # Insert in batch
                except Exception as e:
                    print(f"Insert many objects exception: {e}")

                # Clear the buffer
                objects_buffer = []

                # Check for failed inserts
                if batch_objects_return.has_errors:
                    errors: Dict[int, ErrorObject] = batch_objects_return.errors
                    print(f"Failed to upload {len(errors)} objects")

                    for error_object in errors:
                        print(
                            f"Failed to upload object with error: {error_object.message}"
                        )
                    # TODO: Handle errors
        # TODO: Check buffer is empty, JSON dump failed_objects

    except Exception as e:
        print(f"client operation failed: {e}")

    print(
        "Closing client connection"
    )  # The connection is closed automatically when the context manager exits
