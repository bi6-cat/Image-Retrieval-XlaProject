#!/usr/bin/env python3
"""
Index images to Weaviate with multiple CLIP models for comparison.
This is a SIMPLE version that only encodes images without VQA metadata extraction.
For full metadata extraction, use: python -m app.indexer --model-key <model>

Usage:
    python scripts/index_weaviate_multimodel.py --model clip-base-p32
    python scripts/index_weaviate_multimodel.py --model clip-base-p16
    python scripts/index_weaviate_multimodel.py --model all
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.encoder import Encoder
from app.weaviate_client import get_weaviate_client, create_schema_if_not_exists, batch_add_objects
from app.utils import logger, l2norm_np
from app.indexer import SPECIES_KNOWLEDGE, enrich_metadata_with_knowledge

# Data directory
DATA_DIR = Path("data/full")

def get_all_images(data_dir: Path):
    """Get all image files from data directory"""
    images = []
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    
    for species_dir in data_dir.iterdir():
        if species_dir.is_dir():
            species = species_dir.name
            for img_file in species_dir.iterdir():
                if img_file.suffix.lower() in extensions:
                    images.append({
                        'path': img_file,
                        'species': species,
                        'relative_path': f"full/{species}/{img_file.name}"
                    })
    
    return images

def encode_images_batch(encoder, image_paths, batch_size=16):
    """Encode images in batches"""
    vectors = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                batch_images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
                # Use a blank image as placeholder
                batch_images.append(Image.new('RGB', (224, 224), color='gray'))
        
        # Encode batch
        batch_vectors = encoder.encode_images(batch_images)
        
        # Normalize
        for vec in batch_vectors:
            vectors.append(l2norm_np(vec))
    
    return vectors

def index_model(model_key: str, images: list, batch_size: int = 64):
    """Index images for a specific model"""
    if model_key not in settings.AVAILABLE_MODELS:
        logger.error(f"Unknown model: {model_key}")
        return False
    
    model_info = settings.AVAILABLE_MODELS[model_key]
    model_id = model_info["model_id"]
    collection_name = model_info["collection"]
    
    logger.info(f"=" * 60)
    logger.info(f"Indexing with model: {model_key}")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Total images: {len(images)}")
    logger.info(f"=" * 60)
    
    # Create encoder
    logger.info("Loading encoder...")
    encoder = Encoder(model_name=model_id)
    
    # Create collection if not exists
    logger.info("Creating Weaviate collection if not exists...")
    create_schema_if_not_exists(collection_name)
    
    # Encode all images
    logger.info("Encoding images...")
    image_paths = [img['path'] for img in images]
    vectors = encode_images_batch(encoder, image_paths, batch_size=16)
    
    # Connect to Weaviate
    client = get_weaviate_client()
    
    try:
        # Check current count
        collection = client.collections.get(collection_name)
        current_count = collection.aggregate.over_all(total_count=True).total_count
        logger.info(f"Current objects in collection: {current_count}")
        
        if current_count > 0:
            response = input(f"Collection already has {current_count} objects. Delete and re-index? (y/n): ")
            if response.lower() == 'y':
                logger.info("Deleting existing objects...")
                client.collections.delete(collection_name)
                create_schema_if_not_exists(collection_name)
            else:
                logger.info("Skipping indexing for this model")
                return True
        
        # Prepare objects for batch insert
        logger.info("Preparing objects for batch insert...")
        objects = []
        for i, (img, vector) in enumerate(zip(images, vectors)):
            species = img['species']
            # Enrich with species knowledge
            extra_data = enrich_metadata_with_knowledge(species, {})
            
            # Create meaningful caption from species knowledge
            knowledge = SPECIES_KNOWLEDGE.get(species, {})
            species_name = species.replace('_', ' ')
            distinctive = knowledge.get('distinctive_features', '')
            caption = f"A {species_name}"
            if distinctive:
                caption += f" with {distinctive.split(',')[0]}"  # First feature
            
            obj = {
                "properties": {
                    "file": img['relative_path'],
                    "caption": caption,
                    "species": species,
                    "extra": json.dumps(extra_data),
                    "model_key": model_key
                },
                "vector": vector.tolist()
            }
            objects.append(obj)
        
        # Batch insert
        logger.info(f"Inserting {len(objects)} objects in batches of {batch_size}...")
        for i in tqdm(range(0, len(objects), batch_size), desc="Inserting batches"):
            batch = objects[i:i+batch_size]
            batch_add_objects(client, batch, batch_size=batch_size, collection_name=collection_name)
        
        # Verify
        final_count = collection.aggregate.over_all(total_count=True).total_count
        logger.info(f"✓ Indexing complete! Total objects: {final_count}")
        
    finally:
        client.close()
    
    # Clear GPU memory
    del encoder
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Index images to Weaviate with CLIP models")
    parser.add_argument("--model", type=str, required=True, 
                        help="Model key to use (clip-base-p32, clip-base-p16, or 'all')")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for Weaviate insert")
    args = parser.parse_args()
    
    # Get all images
    logger.info("Scanning for images...")
    images = get_all_images(DATA_DIR)
    logger.info(f"Found {len(images)} images")
    
    if len(images) == 0:
        logger.error("No images found!")
        return
    
    # Print species distribution
    species_count = {}
    for img in images:
        species = img['species']
        species_count[species] = species_count.get(species, 0) + 1
    
    logger.info("Species distribution:")
    for species, count in sorted(species_count.items()):
        logger.info(f"  {species}: {count}")
    
    # Index
    if args.model == "all":
        for model_key in settings.AVAILABLE_MODELS.keys():
            index_model(model_key, images, args.batch_size)
    else:
        index_model(args.model, images, args.batch_size)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
