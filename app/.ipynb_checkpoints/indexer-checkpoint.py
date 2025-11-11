import argparse, json
from pathlib import Path
from app.extractor import FeatureExtractor
from app.weaviate_client import get_weaviate_client, create_schema_if_not_exists, batch_add_objects
from app.config import settings
from app.utils import logger
from tqdm import tqdm
import numpy as np
import uuid

METADATA_QUESTIONS = [
    "What is the main animal?",
    "Is it day or night?",
    "Is there a person in the image?"
]

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

def index_folder(root_folder, weaviate_mode=True, dry_run=True, limit=None):
    extractor = FeatureExtractor()
    files = gather_images(root_folder)
    if limit:
        files = files[:limit]
    logger.info("Found %d files", len(files))

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
        meta_answers = extractor.extract_metadata(pil, METADATA_QUESTIONS)
        species = infer_species(p, Path(root_folder))
        props = {
            "file": str(p),
            "caption": meta_answers.get(METADATA_QUESTIONS[0], ""),
            "species": species or meta_answers.get(METADATA_QUESTIONS[0], ""),
            "extra": json.dumps(meta_answers)
        }
        vec = extractor.encode_image([pil])[0]
        vec = vec.tolist()
        # use seq id or a stable id
        obj_uuid = f"img_{uuid.uuid4().hex[:12]}"  # or f"img_{counter}" 
        batch.append({"uuid": obj_uuid, "properties": props, "vector": vec})


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
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--weaviate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    index_folder(args.data_folder, weaviate_mode=args.weaviate, dry_run=args.dry_run, limit=args.limit)
