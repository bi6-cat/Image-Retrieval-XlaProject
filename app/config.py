from dotenv import load_dotenv
import os
load_dotenv()

class Settings:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))

    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")

    VECTOR_DIM = int(os.getenv("VECTOR_DIM", "512"))
    ENCODER_BACKEND = os.getenv("ENCODER_BACKEND", "hf_clip")

    BETA = float(os.getenv("BETA", 0.1))

    INDEX_BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", 64))
    ENCODE_BATCH_SIZE = int(os.getenv("ENCODE_BATCH_SIZE", 16))
    WEAVIATE_BATCH_SIZE = int(os.getenv("WEAVIATE_BATCH_SIZE", 64))

settings = Settings()
