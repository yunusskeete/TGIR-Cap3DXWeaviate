import weaviate
import weaviate.classes.query as wq
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.connect import ConnectionParams

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
    assert (
        client.is_live()
    ), "Weaviate client is not live"  # This will raise an exception if the client is not live
    print("Client connection established")

    cap3d = client.collections.get(COLLECTION_NAME)
    cap3d_object_count = cap3d.aggregate.over_all(total_count=True).total_count
    cap3d_upload = client.collections.get(DATA_UPLOAD_COLLECTION_NAME)
    cap3d_upload_image_count = cap3d_upload.aggregate.over_all(
        total_count=True
    ).total_count

    print(f"Number of objects in {COLLECTION_NAME} collection: {cap3d_object_count}")
    print(
        f"Number of objects in {DATA_UPLOAD_COLLECTION_NAME} collection: {cap3d_upload_image_count}"
    )

    # Perform query
    response = cap3d.query.near_text(
        query="a chair",
        limit=5,
        return_metadata=wq.MetadataQuery(distance=True),
        # return_properties=[
        #     "description",
        #     "datasetUID",
        # ],
    )

    # Inspect the response
    for o in response.objects:
        print(
            o.properties["description"],
            o.properties["datasetUID"],
        )
        print(
            f"Distance to query: {o.metadata.distance:.3f}\n"
        )  # Print the distance of the object from the query

    uuids = [
        "0b4bb396cb634b248ae9e16653d048e7",
        "46c932c1-d44e-5b30-95d7-4c8909d02fb8",
        "91faaa0e-6a0f-5587-a2dd-fd6e2f3ca3e8",
        "db04848b-c5dd-5ec1-a340-adcc209e9b5e",
        "e574ddc1-3171-5a81-a3aa-d4e2928b9f54",
        "baa41176-79e8-5fbb-8b16-7444f228f88a",
    ]
    obj = cap3d.query.fetch_object_by_id(uuids[0], include_vector=True)
    assert obj, "obj is None"
