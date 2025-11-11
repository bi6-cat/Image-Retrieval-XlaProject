"""
app/redis_utils.py
Helpers để lấy Redis client và lưu/đọc JSON.
"""
import redis
import json
from app.config import settings

_redis_client = None

def get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )
    return _redis_client

def redis_set_json(r, key, obj):
    """
    Lưu Python object dưới dạng JSON string tại key.
    r: redis.Redis instance
    """
    r.set(key, json.dumps(obj))

def redis_get_json(r, key):
    """
    Lấy JSON từ redis và return Python object. Nếu key không tồn tại trả về None.
    """
    v = r.get(key)
    if not v:
        return None
    try:
        return json.loads(v)
    except Exception:
        return v
