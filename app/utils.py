import numpy as np
import logging
import time
from functools import wraps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("image-retrieval")

def l2norm_np(v: np.ndarray):
    v = v.astype("float32")
    n = np.linalg.norm(v) + 1e-12
    return v / n

def retry(exceptions, tries=3, delay=1.0, backoff=2.0):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    logger.warning("Exception: %s — retry in %.1fs (%d tries left)", e, mdelay, mtries-1)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return wrapped
    return decorator
