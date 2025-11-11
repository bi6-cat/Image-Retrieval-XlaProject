import weaviate
from app.config import settings
from app.utils import logger, retry

CLASS_NAME = "AnimalImage"

def get_weaviate_client():
    url = settings.WEAVIATE_URL
    if not url:
        raise RuntimeError("WEAVIATE_URL not set in .env")
    if settings.WEAVIATE_API_KEY:
        auth = weaviate.AuthApiKey(api_key=settings.WEAVIATE_API_KEY)
        client = weaviate.Client(url=url, auth_client_secret=auth)
    else:
        client = weaviate.Client(url=url)
    return client

def create_schema_if_not_exists():
    client = get_weaviate_client()
    schema = {
        "class": CLASS_NAME,
        "vectorizer": "none",
        "properties": [
            {"name": "file", "dataType": ["string"]},
            {"name": "caption", "dataType": ["string"]},
            {"name": "species", "dataType": ["string"]},
            {"name": "extra", "dataType": ["string"]}
        ]
    }
    existing = client.schema.get()
    classes = [c["class"] for c in existing.get("classes", [])]
    if CLASS_NAME not in classes:
        client.schema.create_class(schema)
        logger.info("Weaviate class created: %s", CLASS_NAME)
    else:
        logger.info("Weaviate class exists: %s", CLASS_NAME)

@retry(Exception, tries=3, delay=1.0)
def batch_add_objects(client, objects, batch_size=64):
    with client.batch as batch:
        batch.batch_size = batch_size
        for obj in objects:
            uuid = obj.get("uuid", None)
            if uuid:
                batch.add_data_object(obj["properties"], CLASS_NAME, vector=obj["vector"], uuid=uuid)
            else:
                batch.add_data_object(obj["properties"], CLASS_NAME, vector=obj["vector"])
    logger.info("Batched %d objects to weaviate", len(objects))
