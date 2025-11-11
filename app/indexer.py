import argparse, json
from pathlib import Path
from app.extractor import FeatureExtractor
from app.weaviate_client import get_weaviate_client, create_schema_if_not_exists, batch_add_objects
from app.config import settings
from app.utils import logger
from tqdm import tqdm
import numpy as np
import uuid

METADATA_QUESTIONS = {
    # Basic identification
    "species": "What animal is this? Answer only the species name.",
    "age_group": "Is this a baby or adult animal?",
    
    # Physical appearance
    "color_primary": "What is the main/dominant color of the animal?",
    "size": "Is this animal small, medium, large in size?",
    
    # Behavior & action
    "action": "What is the animal doing? Be specific: sleeping, running, eating, playing, sitting, standing, swimming, flying, etc.",
    "pose": "Describe the pose: lying down, standing, crouching, jumping, profile view, face close-up, etc.",
    
    # Environment & context
    "environment": "Where is this animal? Indoor, outdoor, forest, grassland, desert, snow, water, house, cage",
    "colour-environment": "Describe the main colour environment:",
    "lighting": "Describe the lighting: day, night",
    
    # Additional details
    "interaction": "Is the animal interacting with anything or anyone? Alone, with humans, with other animals, with what dominant objects?",
}

def gather_images(root):
    p = Path(root)
    files = [x for x in p.rglob("*") if x.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    return files

def infer_species(path, root):
    try:
        rel = path.relative_to(root)
        return rel.parts[0] if len(rel.parts) >= 2 else ""
    except Exception:
        return ""

def index_folder(root_folder, weaviate_mode=True, dry_run=True, limit=None, detailed_metadata=False):

    extractor = FeatureExtractor()
    files = gather_images(root_folder)
    if limit:
        files = files[:limit]
    logger.info("Found %d files", len(files))
    
    # Select questions based on detailed_metadata flag
    if detailed_metadata:
        questions = METADATA_QUESTIONS
        logger.info("Using DETAILED metadata extraction (10 fields)")
    else:
        questions = {
            "species": METADATA_QUESTIONS["species"],
            "color_primary": METADATA_QUESTIONS["color_primary"],
            "action": METADATA_QUESTIONS["action"],
            "environment": METADATA_QUESTIONS["environment"],
            "lighting": METADATA_QUESTIONS["lighting"]
        }
        logger.info("Using BASIC metadata extraction (5 fields)")

    if weaviate_mode:
        client = get_weaviate_client()
        create_schema_if_not_exists()
    else:
        client = None

    batch = []
    for p in tqdm(files, desc="Indexing"):
        pil = extractor.load_image(str(p))
        if pil is None:
            continue
        meta_answers = extractor.extract_metadata(pil, questions)
        
        # Get species from folder name
        folder_species = infer_species(p, Path(root_folder))
        species_question = questions["species"]
        vqa_species = meta_answers.get(species_question, "")
        
        # Add folder name to metadata
        meta_answers["folder_name"] = folder_species
        
        props = {
            "file": str(p),
            "caption": meta_answers.get(species_question, ""),
            "species": folder_species or vqa_species,  # Prioritize folder name
            "extra": json.dumps(meta_answers)
        }
        vec = extractor.encode_image([pil])[0]
        vec = vec.tolist()
        # use seq id or a stable id
        
        batch.append({"properties": props, "vector": vec})


        if len(batch) >= settings.WEAVIATE_BATCH_SIZE:
            if not dry_run and weaviate_mode:
                batch_add_objects(client, batch, batch_size=settings.WEAVIATE_BATCH_SIZE)
            batch = []

    if batch:
        if not dry_run and weaviate_mode:
            batch_add_objects(client, batch, batch_size=settings.WEAVIATE_BATCH_SIZE)
    logger.info("Indexing finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", required=True, help="Path to image folder")
    parser.add_argument("--weaviate", action="store_true", help="Use Weaviate for indexing")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually save to database")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--detailed-metadata", "--detailed", action="store_true", 
                        help="Extract detailed metadata (10 fields, slower). Default: basic 5 fields")
    args = parser.parse_args()
    index_folder(
        args.data_folder, 
        weaviate_mode=args.weaviate, 
        dry_run=args.dry_run, 
        limit=args.limit,
        detailed_metadata=args.detailed_metadata
    )
