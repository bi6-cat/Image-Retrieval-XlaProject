# app/api.py  (replace existing)
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np, json, os
from PIL import Image
import io
from datetime import datetime
from typing import Optional, List
from app.config import settings
from app.utils import logger, l2norm_np
from app.encoder import Encoder
from app.weaviate_client import get_weaviate_client, CLASS_NAME, get_collection_name
from app.deps import get_redis, redis_set_json, redis_get_json
import weaviate

# Cache for model encoders - keep all models loaded
model_encoders = {}

def get_encoder_for_model(model_key):
    """Get or create encoder for specific model (keep all models loaded)"""
    if model_key not in settings.AVAILABLE_MODELS:
        model_key = "clip-base-p32"  # fallback to default
    
    # If already loaded, return cached
    if model_key in model_encoders:
        return model_encoders[model_key]
    
    # Load the requested model (keep existing models in memory)
    model_id = settings.AVAILABLE_MODELS[model_key]["model_id"]
    logger.info(f"Loading encoder for {model_key}: {model_id}")
    model_encoders[model_key] = Encoder(model_name=model_id)
    logger.info(f"✓ Loaded encoder for {model_key}. Total models loaded: {len(model_encoders)}")
    
    return model_encoders[model_key]

app = FastAPI(title="Image Retrieval (Multi-Model)")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default encoder (lazy load on first use)
encoder = None
client = None
try:
    client = get_weaviate_client()
    logger.info("Weaviate client ready")
except Exception as e:
    logger.warning("Weaviate client not configured: %s", e)

# DON'T preload all models - use lazy loading to save GPU memory
logger.info("Models will be loaded on demand (lazy loading to save GPU memory)")

# Redis for session
r = get_redis()

@app.on_event("shutdown")
def shutdown_event():
    if client:
        client.close()

class SearchRequest(BaseModel):
    session_id: str
    query_text: str = None
    query_image_vector: list = None
    top_k: int = 20
    user_id: Optional[str] = "anonymous"
    model_key: Optional[str] = "clip-base-p32"  # default model

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

    # Get model-specific encoder and collection
    model_encoder = get_encoder_for_model(req.model_key)
    collection_name = get_collection_name(req.model_key)
    logger.info(f"Search with model: {req.model_key}, collection: {collection_name}")

    if req.query_image_vector is not None:
        qv = np.asarray(req.query_image_vector, dtype=np.float32)
    elif req.query_text:
        qv = model_encoder.encode_text([req.query_text])[0]
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
        import weaviate.classes as wvc
        collection = client.collections.get(collection_name)
        
        # Build query with optional species filter
        if species_filter:
            logger.info(f"Applying species filter: {species_filter}")
            response = collection.query.near_vector(
                near_vector=qv.tolist(),
                limit=req.top_k,
                return_metadata=wvc.query.MetadataQuery(certainty=True, distance=True),
                filters=wvc.query.Filter.by_property("species").equal(species_filter)
            )
        else:
            response = collection.query.near_vector(
                near_vector=qv.tolist(),
                limit=req.top_k,
                return_metadata=wvc.query.MetadataQuery(certainty=True, distance=True)
            )
        
        results = []
        for obj in response.objects:
            results.append({
                "id": str(obj.uuid),
                "score": obj.metadata.certainty if obj.metadata.certainty else (1 - obj.metadata.distance) if obj.metadata.distance else 0.0,
                "meta": { 
                    "file": obj.properties.get("file"), 
                    "caption": obj.properties.get("caption"), 
                    "species": obj.properties.get("species") 
                }
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
    model_key: Optional[str] = "clip-base-p32"

class FeedbackResponse(BaseModel):
    results: list
    refined_vector: list
    turn_feedback_vector: list

@app.post("/search-by-image", response_model=SearchResponse)
async def search_by_image(
    session_id: str = Form(...), 
    file: UploadFile = File(...), 
    top_k: int = Form(20),
    user_id: str = Form("anonymous"),
    model_key: str = Form("clip-base-p32")
):
    """Search by uploading an image"""
    if client is None:
        raise HTTPException(status_code=500, detail="Weaviate client not configured")
    
    # Get model-specific encoder and collection
    model_encoder = get_encoder_for_model(model_key)
    collection_name = get_collection_name(model_key)
    
    try:
        # Read and process uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Encode image to vector
        qv = model_encoder.encode_images([image])[0]
        qv = l2norm_np(qv)
        
        # Search in Weaviate
        import weaviate.classes as wvc
        collection = client.collections.get(collection_name)
        response = collection.query.near_vector(
            near_vector=qv.tolist(),
            limit=top_k,
            return_metadata=wvc.query.MetadataQuery(certainty=True, distance=True)
        )
        
        results = []
        for obj in response.objects:
            results.append({
                "id": str(obj.uuid),
                "score": obj.metadata.certainty if obj.metadata.certainty else (1 - obj.metadata.distance) if obj.metadata.distance else 0.0,
                "meta": { 
                    "file": obj.properties.get("file"), 
                    "caption": obj.properties.get("caption"), 
                    "species": obj.properties.get("species") 
                }
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

    # Get model-specific encoder and collection
    model_encoder = get_encoder_for_model(req.model_key)
    collection_name = get_collection_name(req.model_key)

    # Log request for debugging
    logger.info(f"Feedback request - session: {req.session_id}, model: {req.model_key}, liked: {len(req.liked_image_ids)}, disliked: {len(req.disliked_image_ids)}, text: {bool(req.feedback_text and req.feedback_text.strip())}")

    # Validate at least some feedback
    has_text_feedback = req.feedback_text and req.feedback_text.strip()
    if not req.liked_image_ids and not req.disliked_image_ids and not has_text_feedback:
        raise HTTPException(status_code=400, detail="Please provide at least one feedback: liked images, disliked images, or text")

    # get prev vector
    prev = redis_get_json(r, f"{req.session_id}:previous_search_vector")
    if prev is None:
        raise HTTPException(status_code=400, detail="No previous search found. Please search first.")
    prev_v = np.asarray(prev, dtype=np.float32)

    # encode text feedback
    v_text = None
    if has_text_feedback:
        v_text = model_encoder.encode_text([req.feedback_text])[0]

    # fetch liked image vectors from Weaviate by uuid
    v_like = None
    liked_species = set()  # Track species of liked images
    if req.liked_image_ids:
        vecs = []
        collection = client.collections.get(collection_name)
        for uid in req.liked_image_ids:
            if not uid:  # Skip empty/None IDs
                continue
            try:
                obj = collection.query.fetch_object_by_id(uid, include_vector=True)
                if obj and obj.vector:
                    vecs.append(np.array(obj.vector["default"] if isinstance(obj.vector, dict) else obj.vector, dtype=np.float32))
                    # Track species of liked images
                    if obj.properties.get("species"):
                        liked_species.add(obj.properties.get("species"))
            except Exception as e:
                logger.warning("Failed to fetch liked object %s: %s", uid, e)
        if vecs:
            v_like = np.mean(np.stack(vecs, axis=0), axis=0)

    # fetch disliked image vectors from Weaviate by uuid
    v_dislike = None
    disliked_species = set()  # Track species of disliked images
    if req.disliked_image_ids:
        vecs = []
        collection = client.collections.get(collection_name)
        for uid in req.disliked_image_ids:
            if not uid:  # Skip empty/None IDs
                continue
            try:
                obj = collection.query.fetch_object_by_id(uid, include_vector=True)
                if obj and obj.vector:
                    vecs.append(np.array(obj.vector["default"] if isinstance(obj.vector, dict) else obj.vector, dtype=np.float32))
                    # Track species of disliked images
                    if obj.properties.get("species"):
                        disliked_species.add(obj.properties.get("species"))
            except Exception as e:
                logger.warning("Failed to fetch disliked object %s: %s", uid, e)
        if vecs:
            v_dislike = np.mean(np.stack(vecs, axis=0), axis=0)
    
    logger.info(f"Liked species: {liked_species}, Disliked species: {disliked_species}")

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
    
    # NOTE: For negative feedback, we DON'T modify the vector anymore
    # Instead, we use filtering:
    # 1. Keep same species as liked images (filter in Weaviate query)
    # 2. Exclude disliked images from results (post-filter)
    # This approach works better because CLIP embeddings don't cleanly separate
    # "species" from "secondary attributes" (color, pose, etc.)
    if v_dislike is not None:
        logger.info("Negative feedback: will exclude %d disliked images from results (keeping species filter)", len(req.disliked_image_ids))

    # Ensure v_turn is set for response (even if None, set to v_new)
    if v_turn is None:
        v_turn = v_new

    # update previous vector in redis
    redis_set_json(r, f"{req.session_id}:previous_search_vector", v_new.tolist())
    # optionally update aggregated vector / history similar to earlier design

    # search again in Weaviate
    try:
        import weaviate.classes as wvc
        collection = client.collections.get(collection_name)
        
        # Build filter: prioritize liked species, exclude disliked images
        search_filter = None
        
        # If we have liked species, filter to only show same species
        if liked_species:
            # Filter to only show images from liked species
            species_filters = [
                wvc.query.Filter.by_property("species").equal(sp) 
                for sp in liked_species
            ]
            if len(species_filters) == 1:
                search_filter = species_filters[0]
            else:
                search_filter = wvc.query.Filter.any_of(species_filters)
            logger.info(f"Filtering results to species: {liked_species}")
        
        # Request more results to filter out disliked ones
        fetch_limit = req.top_k + len(req.disliked_image_ids) * 2
        
        response = collection.query.near_vector(
            near_vector=v_new.tolist(),
            limit=fetch_limit,
            filters=search_filter,
            return_metadata=wvc.query.MetadataQuery(certainty=True, distance=True)
        )
        
        # Filter out disliked images from results
        disliked_set = set(req.disliked_image_ids)
        results = []
        for obj in response.objects:
            obj_id = str(obj.uuid)
            # Skip disliked images
            if obj_id in disliked_set:
                continue
            results.append({
                "id": obj_id,
                "score": obj.metadata.certainty if obj.metadata.certainty else (1 - obj.metadata.distance) if obj.metadata.distance else 0.0,
                "meta": { 
                    "file": obj.properties.get("file"), 
                    "caption": obj.properties.get("caption"), 
                    "species": obj.properties.get("species")
                }
            })
            # Stop when we have enough results
            if len(results) >= req.top_k:
                break
                
        logger.info(f"Returning {len(results)} results after filtering")
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
        "encoder_ready": True,  # Lazy loading - always ready
        "loaded_models": list(model_encoders.keys())
    }

@app.get("/models")
def get_available_models():
    """Get list of available models"""
    models = []
    for key, info in settings.AVAILABLE_MODELS.items():
        models.append({
            "key": key,
            "name": info["name"],
            "description": info["description"],
            "model_id": info["model_id"],
            "collection": info["collection"]
        })
    return {"models": models, "default": "clip-base-p32"}

@app.get("/stats")
def get_stats():
    """Get database statistics for all models"""
    if client is None:
        raise HTTPException(status_code=500, detail="Weaviate not configured")
    
    stats = {}
    total = 0
    
    # Get stats for each model's collection
    for model_key, model_info in settings.AVAILABLE_MODELS.items():
        collection_name = model_info["collection"]
        try:
            collection = client.collections.get(collection_name)
            response = collection.aggregate.over_all(total_count=True)
            count = response.total_count
            stats[model_key] = {
                "collection": collection_name,
                "count": count,
                "model_name": model_info["name"]
            }
            total += count
        except Exception as e:
            logger.warning(f"Stats error for {model_key}: %s", e)
            stats[model_key] = {
                "collection": collection_name,
                "count": 0,
                "error": str(e)
            }
    
    return {
        "total_images": total,
        "models": stats
    }

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

@app.post("/upload-image")
async def upload_image_to_db(
    file: UploadFile = File(...),
    caption: str = Form(""),
    species: str = Form(""),
    user_id: str = Form("anonymous")
):
    """Upload and index a new image to clip-base-p32 model only"""
    import time
    start_time = time.time()
    
    if client is None:
        raise HTTPException(status_code=500, detail="Weaviate client not configured")
    
    try:
        # Read and validate image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        read_time = time.time() - start_time
        
        # Generate unique filename
        import uuid
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ext = os.path.splitext(file.filename)[1] or '.jpg'
        unique_filename = f"user_upload_{timestamp}_{uuid.uuid4().hex[:8]}{file_ext}"
        
        # Determine species from caption or default
        if not species:
            species = "uploaded"  # default category
        
        # Save image to data directory
        upload_dir = f"data/full/{species}"
        os.makedirs(upload_dir, exist_ok=True)
        image_path = os.path.join(upload_dir, unique_filename)
        image.save(image_path)
        save_time = time.time() - start_time - read_time
        
        relative_path = f"full/{species}/{unique_filename}"
        
        # Index to clip-base-p32 only
        import weaviate.classes as wvc
        model_key = "clip-base-p32"
        model_info = settings.AVAILABLE_MODELS[model_key]
        
        # Get encoder for this model
        encode_start = time.time()
        model_encoder = get_encoder_for_model(model_key)
        collection_name = model_info["collection"]
        
        # Encode image
        vector = model_encoder.encode_images([image])[0]
        vector = l2norm_np(vector)
        encode_time = time.time() - encode_start
        
        # Prepare object
        obj_data = {
            "properties": {
                "file": relative_path,
                "caption": caption or f"User uploaded image",
                "species": species,
                "extra": json.dumps({"uploaded_by": user_id, "uploaded_at": timestamp}),
                "model_key": model_key
            },
            "vector": vector.tolist()
        }
        
        # Insert to Weaviate
        db_start = time.time()
        collection = client.collections.get(collection_name)
        uuid_result = collection.data.insert(
            properties=obj_data["properties"],
            vector=obj_data["vector"]
        )
        db_time = time.time() - db_start
        
        total_time = time.time() - start_time
        
        logger.info(f"Upload completed in {total_time:.2f}s - Read: {read_time:.2f}s, Save: {save_time:.2f}s, Encode: {encode_time:.2f}s, DB: {db_time:.2f}s")
        
        return {
            "success": True,
            "message": "Image uploaded and indexed successfully",
            "image_path": relative_path,
            "collection": collection_name,
            "uuid": str(uuid_result),
            "species": species,
            "processing_time": {
                "total": round(total_time, 2),
                "read_image": round(read_time, 2),
                "save_file": round(save_time, 2),
                "encode_vector": round(encode_time, 2),
                "database_insert": round(db_time, 2)
            }
        }
        
    except Exception as e:
        logger.exception("Upload error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
