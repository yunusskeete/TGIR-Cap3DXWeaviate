import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import weaviate
import weaviate.classes as wvc
import weaviate.classes.config as wc
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
BUFFER_SIZE = 500
PATH_TO_EXAMPLE_OBJECTS = "/home/yunusskeete/Documents/data/3D/Cap3D/local-split/unzips/compressed_imgs_perobj_00.zip/Cap3D_Objaverse_renderimgs"
IMAGE_FILE_EXTENSION = ".png"
IMAGE_FILE_DELIMETER = "_"
IMAGE_ENCODING = "utf-8"
DELETE_ON_UPLOAD = False

PERFORMING_CHECKSUM = False

path_to_example_objects = Path(PATH_TO_EXAMPLE_OBJECTS)

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
        cap3d_upload: Collection = create_collection(
            client=client,
            collection_name=DATA_UPLOAD_COLLECTION_NAME,
            configure_upload_collection=True,
        )
        assert cap3d_upload.aggregate.over_all(total_count=True).total_count == 0

        objects_buffer: List[DataObject] = []
        failed_objects: List[Tuple[int, str]] = []

        for object_idx, object_folder in tqdm(
            enumerate(path_to_example_objects.iterdir())
        ):
            # if object_idx < 2191:  # 1264:
            #     continue
            image_uuids_per_object: List[str] = []
            image_objects_buffer: List[DataObject] = []

            object_description: str = descriptions_dict.get(
                object_folder.name, ""
            )  # TODO: Add log if object could not be found, add to tracking list and DO NOT UPLOAD

            for object_image_idx, object_image_file in enumerate(
                sorted(
                    (
                        file
                        for file in object_folder.iterdir()
                        if file.suffix == IMAGE_FILE_EXTENSION
                        and IMAGE_FILE_DELIMETER not in file.name
                    ),
                    key=lambda file: file.name,
                )
            ):
                # Convert image to base64
                with object_image_file.open("rb") as file:
                    image_b64: str = base64.b64encode(file.read()).decode(
                        IMAGE_ENCODING
                    )

                # Build the image object payload
                image_obj: Dict[str, str] = {
                    "image": image_b64,
                    "description": object_description,
                    "datasetUID": f"{object_image_file.parent.name}_{object_image_file.name}",  # E.g. "c5517f31ede34ad0a0da1f38753f9588_00005.png"
                }

                image_object_uuid: str = generate_uuid5(image_obj["datasetUID"])

                image_uuids_per_object.append(image_object_uuid)

                image_objects_buffer.append(
                    DataObject(
                        properties=image_obj,
                        uuid=image_object_uuid,
                    )
                )  # Batcher automatically sends batches

            try:
                batch_image_objects_return: BatchObjectReturn = (
                    cap3d_upload.data.insert_many(
                        image_objects_buffer
                    )  # Insert in batch
                )
            except weaviate.exceptions.WeaviateInsertManyAllFailedError as e:
                print(e)
                failed_object_identifier: Tuple[int, str] = (
                    object_idx,
                    object_folder.name,
                )
                failed_objects.append(failed_object_identifier)
                print(failed_object_identifier)
                continue

            data_objects: List[
                weaviate.collections.classes.internal.ObjectSingleReturn
            ] = [
                cap3d_upload.query.fetch_object_by_id(uuid, include_vector=True)
                for uuid in image_uuids_per_object
            ]
            data_objects = [obj for obj in data_objects if obj]
            assert len(data_objects) == 20, "Missing data object"

            try:
                average_vector: List[float] = np.mean(
                    np.array(
                        [data_object.vector["default"] for data_object in data_objects]
                    ),
                    axis=0,
                ).tolist()
            except Exception as e:
                print(f"Numpy exception: {e}")

            assert isinstance(
                average_vector, list
            ), f"Vector is of invalid format: received '{type(average_vector)}', expecting '{type([])}'"

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
                        vector=average_vector,
                        uuid=object_uuid,
                    )
                )
            except Exception as e:
                print(e)
                failed_object_identifier: Tuple[int, str] = (
                    object_idx,
                    object_folder.name,
                )
                failed_objects.append(failed_object_identifier)
                print(failed_object_identifier)
                print(f"DataObject instantiation exception: {e}")

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

            if DELETE_ON_UPLOAD:
                # On success, delete image objects from cap3d_upload collection
                cap3d_upload.data.delete_many(
                    where=Filter.by_id().contains_any(image_uuids_per_object)
                )

    except Exception as e:
        print(f"client operation failed: {e}")

    print("Closing client connection")
    pass
    # The connection is closed automatically when the context manager exits