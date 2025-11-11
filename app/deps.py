"""Simple Redis helpers used by api.py.

Provides:
 - get_redis(): return redis.Redis instance
 - redis_set_json(r, key, obj): set JSON-serialised value
 - redis_get_json(r, key): get and parse JSON value
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
    """Store Python object as JSON string in Redis (string key)."""
    r.set(key, json.dumps(obj))

def redis_get_json(r, key):
    v = r.get(key)
    if not v:
        return None
    try:
        return json.loads(v)
    except Exception:
        # if stored as plain string, return raw
        return v
