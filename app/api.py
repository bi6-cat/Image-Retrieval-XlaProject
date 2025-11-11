# app/api.py  (replace existing)
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np, json
from PIL import Image
import io
from datetime import datetime
from typing import Optional, List
from app.config import settings
from app.utils import logger, l2norm_np
from app.encoder import Encoder
from app.weaviate_client import get_weaviate_client, CLASS_NAME
from app.deps import get_redis, redis_set_json, redis_get_json
import weaviate

app = FastAPI(title="Image Retrieval (Weaviate)")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


encoder = Encoder()
client = None
try:
    client = get_weaviate_client()
    logger.info("Weaviate client ready")
except Exception as e:
    logger.warning("Weaviate client not configured: %s", e)
# Redis for session
r = get_redis()

class SearchRequest(BaseModel):
    session_id: str
    query_text: str = None
    query_image_vector: list = None
    top_k: int = 20
    user_id: Optional[str] = "anonymous"

class SearchResponse(BaseModel):
    results: list
    used_vector: list

class SearchHistoryEntry(BaseModel):
    session_id: str
    user_id: str
    query_text: Optional[str] = None
    query_type: str  # "text" or "image"
    timestamp: str
    num_results: int
    top_result_id: Optional[str] = None

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="Weaviate client not configured")

    if req.query_image_vector is not None:
        qv = np.asarray(req.query_image_vector, dtype=np.float32)
    elif req.query_text:
        qv = encoder.encode_text([req.query_text])[0]
    else:
        raise HTTPException(status_code=400, detail="Provide query_text or query_image_vector")
    qv = l2norm_np(qv)

    # Extract species from query for filtering
    query_lower = req.query_text.lower() if req.query_text else ""
    species_filter = None
    
    # Common species keywords
    species_keywords = {
        'cat': 'cat', 'cats': 'cat', 'kitten': 'cat', 'feline': 'cat',
        'dog': 'dog', 'dogs': 'dog', 'puppy': 'dog', 'canine': 'dog',
        'bird': 'bird', 'birds': 'bird',
        'horse': 'horse', 'horses': 'horse',
        'cow': 'cow', 'cows': 'cow', 'cattle': 'cow',
        'sheep': 'sheep',
        'elephant': 'elephant', 'elephants': 'elephant',
        'butterfly': 'butterfly', 'butterflies': 'butterfly'
    }
    
    for keyword, species in species_keywords.items():
        if keyword in query_lower:
            species_filter = species
            break

    # Weaviate nearVector query with optional species filter
    try:
        query_builder = client.query.get(CLASS_NAME, ["file","caption","species","extra"])
        
        # Add species filter if detected
        if species_filter:
            where_filter = {
                "path": ["species"],
                "operator": "Equal",
                "valueString": species_filter
            }
            query_builder = query_builder.with_where(where_filter)
            logger.info(f"Applying species filter: {species_filter}")
        
        res = query_builder.with_near_vector({"vector": qv.tolist()}).with_limit(req.top_k).do()
        objs = res.get("data", {}).get("Get", {}).get(CLASS_NAME, [])
        results = []
        # each obj may have _additional containing id and vector/score
        for o in objs:
            add = o.get("_additional", {})
            results.append({
                "id": add.get("id"),           # this is uuid we assigned
                "score": add.get("certainty") or add.get("distance") or 0.0,
                "meta": { "file": o.get("file"), "caption": o.get("caption"), "species": o.get("species") }
            })
        # store previous_search_vector in redis for session
        redis_set_json(r, f"{req.session_id}:previous_search_vector", qv.tolist())
        
        # Save search history
        save_search_history(
            session_id=req.session_id,
            user_id=getattr(req, 'user_id', 'anonymous'),
            query_text=req.query_text,
            query_type="text",
            num_results=len(results),
            top_result_id=results[0]["id"] if results else None
        )
        
        return SearchResponse(results=results, used_vector=qv.tolist())
    except Exception as e:
        logger.exception("Weaviate query error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# Feedback endpoint using Weaviate
class FeedbackRequest(BaseModel):
    session_id: str
    feedback_text: Optional[str] = None
    liked_image_ids: List[str] = []
    disliked_image_ids: List[str] = []
    w_text: float = 0.5
    w_like: float = 0.5
    alpha: float = 0.4
    gamma: float = 0.5
    top_k: int = 20

class FeedbackResponse(BaseModel):
    results: list
    refined_vector: list
    turn_feedback_vector: list

@app.post("/search-by-image", response_model=SearchResponse)
async def search_by_image(
    session_id: str = Form(...), 
    file: UploadFile = File(...), 
    top_k: int = Form(20),
    user_id: str = Form("anonymous")
):
    """Search by uploading an image"""
    if client is None:
        raise HTTPException(status_code=500, detail="Weaviate client not configured")
    
    try:
        # Read and process uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Encode image to vector
        qv = encoder.encode_images([image])[0]
        qv = l2norm_np(qv)
        
        # Search in Weaviate
        res = client.query.get(CLASS_NAME, ["file","caption","species"]).with_near_vector({"vector": qv.tolist()}).with_limit(top_k).do()
        objs = res.get("data", {}).get("Get", {}).get(CLASS_NAME, [])
        results = []
        for o in objs:
            add = o.get("_additional", {})
            results.append({
                "id": add.get("id"),
                "score": add.get("certainty") or add.get("distance") or 0.0,
                "meta": { "file": o.get("file"), "caption": o.get("caption"), "species": o.get("species") }
            })
        
        # Store vector for feedback
        redis_set_json(r, f"{session_id}:previous_search_vector", qv.tolist())
        
        # Save search history
        save_search_history(
            session_id=session_id,
            user_id=user_id,
            query_text=f"[Image: {file.filename}]",
            query_type="image",
            num_results=len(results),
            top_result_id=results[0]["id"] if results else None
        )
        
        return SearchResponse(results=results, used_vector=qv.tolist())
    except Exception as e:
        logger.exception("Image upload search error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="Weaviate client not configured")

    # Log request for debugging
    logger.info(f"Feedback request - session: {req.session_id}, liked: {len(req.liked_image_ids)}, disliked: {len(req.disliked_image_ids)}, text: {bool(req.feedback_text)}")

    # Validate at least some feedback
    if not req.liked_image_ids and not req.disliked_image_ids and not req.feedback_text:
        raise HTTPException(status_code=400, detail="Please provide at least one feedback: liked images, disliked images, or text")

    # get prev vector
    prev = redis_get_json(r, f"{req.session_id}:previous_search_vector")
    if prev is None:
        raise HTTPException(status_code=400, detail="No previous search found. Please search first.")
    prev_v = np.asarray(prev, dtype=np.float32)

    # encode text feedback
    v_text = None
    if req.feedback_text:
        v_text = encoder.encode_text([req.feedback_text])[0]

    # fetch liked image vectors from Weaviate by uuid
    v_like = None
    if req.liked_image_ids:
        vecs = []
        for uid in req.liked_image_ids:
            if not uid:  # Skip empty/None IDs
                continue
            try:
                obj = client.data_object.get_by_id(uid, with_vector=True)
                if obj and "vector" in obj:
                    vec = obj["vector"]
                    vecs.append(np.array(vec, dtype=np.float32))
            except Exception as e:
                logger.warning("Failed to fetch liked object %s: %s", uid, e)
        if vecs:
            v_like = np.mean(np.stack(vecs, axis=0), axis=0)

    # fetch disliked image vectors from Weaviate by uuid
    v_dislike = None
    if req.disliked_image_ids:
        vecs = []
        for uid in req.disliked_image_ids:
            if not uid:  # Skip empty/None IDs
                continue
            try:
                obj = client.data_object.get_by_id(uid, with_vector=True)
                if obj and "vector" in obj:
                    vec = obj["vector"]
                    vecs.append(np.array(vec, dtype=np.float32))
            except Exception as e:
                logger.warning("Failed to fetch disliked object %s: %s", uid, e)
        if vecs:
            v_dislike = np.mean(np.stack(vecs, axis=0), axis=0)

    # fuse with Rocchio algorithm: Q_new = Q_old + α*relevant - β*non_relevant
    v_turn = None
    if v_text is not None and v_like is not None:
        v_turn = l2norm_np(req.w_text * v_text + req.w_like * v_like)
    elif v_text is not None:
        v_turn = l2norm_np(v_text)
    elif v_like is not None:
        v_turn = l2norm_np(v_like)
    
    # Apply positive feedback
    if v_turn is not None:
        v_new = l2norm_np((1 - req.alpha) * prev_v + req.alpha * v_turn)
    else:
        v_new = prev_v
    
    # Apply negative feedback (move away from disliked images)
    if v_dislike is not None:
        # Subtract disliked vector with smaller weight (gamma)
        v_new = l2norm_np(v_new - req.gamma * v_dislike)
        logger.info("Applied negative feedback with %d disliked images", len(req.disliked_image_ids))

    # Ensure v_turn is set for response (even if None, set to v_new)
    if v_turn is None:
        v_turn = v_new

    # update previous vector in redis
    redis_set_json(r, f"{req.session_id}:previous_search_vector", v_new.tolist())
    # optionally update aggregated vector / history similar to earlier design

    # search again in Weaviate
    try:
        res = client.query.get(CLASS_NAME, ["file","caption","species"]).with_near_vector({"vector": v_new.tolist()}).with_limit(req.top_k).do()
        objs = res.get("data", {}).get("Get", {}).get(CLASS_NAME, [])
        results = []
        for o in objs:
            add = o.get("_additional", {})
            results.append({
                "id": add.get("id"),
                "score": add.get("certainty") or add.get("distance") or 0.0,
                "meta": { "file": o.get("file"), "caption": o.get("caption"), "species": o.get("species")}
            })
        return FeedbackResponse(results=results, refined_vector=v_new.tolist(), turn_feedback_vector=v_turn.tolist())
    except Exception as e:
        logger.exception("Weaviate rerank error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "weaviate": client is not None,
        "redis": r is not None,
        "encoder": encoder is not None
    }

@app.get("/stats")
def get_stats():
    """Get database statistics"""
    if client is None:
        raise HTTPException(status_code=500, detail="Weaviate not configured")
    try:
        result = client.query.aggregate(CLASS_NAME).with_meta_count().do()
        count = result.get("data", {}).get("Aggregate", {}).get(CLASS_NAME, [{}])[0].get("meta", {}).get("count", 0)
        return {
            "total_images": count,
            "collection": CLASS_NAME
        }
    except Exception as e:
        logger.exception("Stats error: %s", e)
        return {"total_images": 0, "error": str(e)}

def save_search_history(session_id: str, user_id: str, query_text: str, query_type: str, num_results: int, top_result_id: str):
    """Save search history to Redis"""
    try:
        history_key = f"history:{user_id}"
        entry = {
            "session_id": session_id,
            "user_id": user_id,
            "query_text": query_text,
            "query_type": query_type,
            "timestamp": datetime.now().isoformat(),
            "num_results": num_results,
            "top_result_id": top_result_id
        }
        # Add to list (keep last 100)
        history = redis_get_json(r, history_key) or []
        history.insert(0, entry)
        history = history[:100]  # Keep only last 100 searches
        redis_set_json(r, history_key, history)
        logger.info(f"Saved search history for user {user_id}")
    except Exception as e:
        logger.warning(f"Failed to save search history: {e}")

@app.get("/history/{user_id}")
def get_search_history(user_id: str, limit: int = 20):
    """Get search history for a user"""
    try:
        history_key = f"history:{user_id}"
        history = redis_get_json(r, history_key) or []
        return {"history": history[:limit]}
    except Exception as e:
        logger.exception("Failed to get history: %s", e)
        return {"history": [], "error": str(e)}

@app.get("/analytics")
def get_analytics():
    """Get analytics across all users"""
    try:
        # Get all history keys
        all_keys = []
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match="history:*", count=100)
            all_keys.extend(keys)
            if cursor == 0:
                break
        
        total_searches = 0
        query_types = {"text": 0, "image": 0}
        top_queries = {}
        
        for key in all_keys:
            history = redis_get_json(r, key) or []
            total_searches += len(history)
            for entry in history:
                query_types[entry.get("query_type", "text")] += 1
                query_text = entry.get("query_text", "")
                if query_text and not query_text.startswith("[Image:"):
                    top_queries[query_text] = top_queries.get(query_text, 0) + 1
        
        # Sort top queries
        sorted_queries = sorted(top_queries.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_searches": total_searches,
            "total_users": len(all_keys),
            "query_types": query_types,
            "top_queries": [{"query": q, "count": c} for q, c in sorted_queries]
        }
    except Exception as e:
        logger.exception("Analytics error: %s", e)
        return {"error": str(e)}

app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
