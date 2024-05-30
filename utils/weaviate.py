from typing import List, Optional

import weaviate
import weaviate.classes as wvc
import weaviate.classes.config as wc
from weaviate import WeaviateClient


def create_collection(
    client: WeaviateClient,
    collection_name: str,
    configure_upload_collection: Optional[bool] = False,
) -> weaviate.collections.Collection:
    """
    Creates a collection in a Weaviate vector database with a given configuration.

    This function creates a Weaviate collection (also known as a "class") in a Weaviate vector
    database. The specific configuration depends on the `configure_upload_collection` parameter.
    If `False`, a vector-only collection is created; if `True`, a data upload collection with
    Multi2Vec-CLIP configurations is created.

    Parameters
    ----------
    client : WeaviateClient
        The Weaviate client used to interact with the Weaviate server.
    collection_name : str
        The name of the collection to be created in Weaviate.
    configure_upload_collection : Optional[bool], default False
        If `True`, creates a collection for data upload with Multi2Vec-CLIP configurations.
        If `False`, creates a vector-only collection with HNSW (Hierarchical Navigable Small World)
        index configuration.

    Returns
    -------
    weaviate.collections.Collection
        The created Weaviate collection with specified properties and configurations.

    Raises
    ------
    weaviate.exceptions.WeaviateBaseError
        If there's an error during collection creation or configuration.

    Notes
    -----
    - The `collection_name` parameter specifies the name of the new Weaviate collection.
    - When `configure_upload_collection` is `False`, the collection includes text-based properties
      like 'description' and 'datasetUID', with HNSW for vector indexing using COSINE distance metric.
    - When `configure_upload_collection` is `True`, the collection is designed for Multi2Vec-CLIP,
      including image-based property 'image' and text-based properties, with specific vectorizer
      configurations for image and text.
    """
    cap3d_properties: List[weaviate.classes.config.Property] = [
        wc.Property(
            name="description",
            data_type=wc.DataType.TEXT,
            description="Description of image",
        ),
        wc.Property(
            name="datasetUID",
            data_type=wc.DataType.TEXT,
            description="A concatenation of the Unique ID and the name of the image from dataset",
        ),
    ]
    cap3d_upload_properties: List[weaviate.classes.config.Property] = [
        wc.Property(
            name="image",
            data_type=wc.DataType.BLOB,  # base64-encoded string
            description="Image",
        ),
        wc.Property(
            name="description",
            data_type=wc.DataType.TEXT,
            description="Description of image",
        ),
        wc.Property(
            name="datasetUID",
            data_type=wc.DataType.TEXT,
            description="A concatenation of the Unique ID and the name of the image from dataset",
        ),
    ]

    cap3d_vectorizer_config: weaviate.classes.config.Configure.Vectorizer = (
        wvc.config.Configure.Vectorizer.none()
    )
    cap3d_upload_vectorizer_config: List[
        weaviate.classes.config.Configure.Vectorizer
    ] = wc.Configure.Vectorizer.multi2vec_clip(
        image_fields=[
            wc.Multi2VecField(name="image", weight=0.9)
        ],  # 90% of the vector is from the poster
        text_fields=[
            wc.Multi2VecField(name="description", weight=0.1)
        ],  # 10% of the vector is from the title),
    )

    vector_index_config: weaviate.classes.config.Configure.VectorIndex = (
        wvc.config.Configure.VectorIndex.hnsw(
            distance_metric=wvc.config.VectorDistances.COSINE  # select prefered distance metric
        )
    )

    return (
        client.collections.create(
            collection_name,
            description="A vector-only Cap3D collection for multi2vec-clip",
            properties=cap3d_properties,
            vectorizer_config=cap3d_vectorizer_config,
            vector_index_config=vector_index_config,
        )
        if not configure_upload_collection
        else client.collections.create(
            collection_name,
            description="A data upload Cap3D collection for multi2vec-clip",
            properties=cap3d_upload_properties,
            vectorizer_config=cap3d_upload_vectorizer_config,
        )
    )
