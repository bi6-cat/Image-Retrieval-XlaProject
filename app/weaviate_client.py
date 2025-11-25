import weaviate
import weaviate.classes as wvc
from app.config import settings
from app.utils import logger, retry

CLASS_NAME = "AnimalImage"  # Legacy default

def get_collection_name(model_key=None):
    """Get collection name for a specific model"""
    if model_key and model_key in settings.AVAILABLE_MODELS:
        return settings.AVAILABLE_MODELS[model_key]["collection"]
    return CLASS_NAME

def get_weaviate_client():
    url = settings.WEAVIATE_URL
    if not url:
        raise RuntimeError("WEAVIATE_URL not set in .env")
    if settings.WEAVIATE_API_KEY:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(settings.WEAVIATE_API_KEY)
        )
    else:
        client = weaviate.connect_to_custom(
            http_host=url.replace("https://", "").replace("http://", ""),
            http_secure=url.startswith("https://")
        )
    return client

def create_schema_if_not_exists(collection_name=None):
    collection_name = collection_name or CLASS_NAME
    client = get_weaviate_client()
    try:
        # Check if collection exists
        if client.collections.exists(collection_name):
            logger.info("Weaviate collection exists: %s", collection_name)
        else:
            # Create collection with properties
            from weaviate.classes.config import Configure, Property, DataType
            client.collections.create(
                name=collection_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="file", data_type=DataType.TEXT),
                    Property(name="caption", data_type=DataType.TEXT),
                    Property(name="species", data_type=DataType.TEXT),
                    Property(name="extra", data_type=DataType.TEXT),
                    Property(name="model_key", data_type=DataType.TEXT)
                ]
            )
            logger.info("Weaviate collection created: %s", collection_name)
    finally:
        client.close()

@retry(Exception, tries=3, delay=1.0)
def batch_add_objects(client, objects, batch_size=64, collection_name=None):
    collection_name = collection_name or CLASS_NAME
    collection = client.collections.get(collection_name)
    with collection.batch.dynamic() as batch:
        for obj in objects:
            properties = obj["properties"]
            vector = obj["vector"]
            uuid = obj.get("uuid", None)
            if uuid:
                batch.add_object(properties=properties, vector=vector, uuid=uuid)
            else:
                batch.add_object(properties=properties, vector=vector)
    logger.info("Batched %d objects to weaviate collection %s", len(objects), collection_name)
