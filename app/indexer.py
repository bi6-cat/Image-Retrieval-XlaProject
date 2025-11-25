import argparse, json
from pathlib import Path
from app.extractor import FeatureExtractor
from app.weaviate_client import get_weaviate_client, create_schema_if_not_exists, batch_add_objects, get_collection_name
from app.config import settings
from app.utils import logger
from tqdm import tqdm
import numpy as np
import uuid
import os

# Optimized metadata questions - focus on visual features only
# Biological knowledge (animal_class, body_coverage, locomotion, diet, habitat) 
# is provided by SPECIES_KNOWLEDGE instead
METADATA_QUESTIONS = {
    # Physical appearance (visual only)
    "color_primary": "What are the main colors of the animal? List 1-3 colors",
    "color_pattern": "What color pattern? solid, striped, spotted, patched, mixed",
    "distinctive_features": "What distinctive features are visible? Examples: long neck, large ears, horns, wings, trunk, tail, beak, tusks",
    
    # Behavior & action (what's happening in image)
    "action": "What is the animal doing? sleeping, running, eating, playing, sitting, standing, swimming, flying, walking, resting",
    "body_posture": "Body posture? standing, sitting, lying down, crouching, stretched",
    
    # Environment & context (image setting)
    "environment": "Where is this? indoor, outdoor, forest, grassland, desert, snow, water, mountain, zoo, house",
    "background": "Background type? natural habitat, plain background, vegetation, water, buildings",
    
    # Objects & interactions in the scene
    "visible_objects": "What objects are visible in the image? Examples: ball, toy, food bowl, leash, furniture, tree, rock, fence, fence, car, person",
    "interaction_with": "What is the animal interacting with? nothing, human, another animal, toy, food, water, furniture, ground",
    
    # Image composition
    "number_of_animals": "How many animals? one, two, few (3-5), several (6+)",
    "camera_angle": "Camera angle? front view, side view, close-up, full body",
}

# Knowledge base for species-specific information (used when species is known)
SPECIES_KNOWLEDGE = {
    "alpaca": {
        "animal_class": "mammal",
        "distinctive_features": "long neck, small head, fluffy wool coat, banana-shaped ears, split upper lip",
        "body_coverage": "thick wool fleece",
        "locomotion_type": "walking on four legs",
        "diet_indication": "herbivore (grazer)",
        "habitat_type": "terrestrial, high altitude grasslands and mountains of South America"
    },
    "bat": {
        "animal_class": "mammal",
        "distinctive_features": "wings made of skin membrane, small body, large ears, echolocation ability, clawed thumbs",
        "body_coverage": "fur with wing membranes",
        "locomotion_type": "flying, hanging upside down",
        "diet_indication": "insectivore or frugivore depending on species",
        "habitat_type": "aerial and arboreal, caves and trees"
    },
    "brown_bear": {
        "animal_class": "mammal",
        "distinctive_features": "large size, muscular hump on shoulders, rounded ears, long claws, brown fur",
        "body_coverage": "thick fur",
        "locomotion_type": "walking on four legs, can stand on hind legs",
        "diet_indication": "omnivore (fish, berries, meat, plants)",
        "habitat_type": "terrestrial, forests and mountains"
    },
    "buffalo": {
        "animal_class": "mammal",
        "distinctive_features": "large curved horns, massive body, thick neck, sturdy legs, dark hide",
        "body_coverage": "thick skin with sparse dark hair",
        "locomotion_type": "walking on four legs",
        "diet_indication": "herbivore (grazer)",
        "habitat_type": "terrestrial, grasslands and savannas"
    },
    "butterfly": {
        "animal_class": "insect",
        "distinctive_features": "colorful wings with patterns, antennae, six legs, proboscis, metamorphosis lifecycle",
        "body_coverage": "exoskeleton with wing scales",
        "locomotion_type": "flying",
        "diet_indication": "herbivore (nectar-feeder)",
        "habitat_type": "terrestrial with aerial movement, gardens and meadows"
    },
    "camel": {
        "animal_class": "mammal",
        "distinctive_features": "one or two humps for fat storage, long neck, thick lips, long eyelashes, padded feet",
        "body_coverage": "short to medium fur",
        "locomotion_type": "walking on four legs",
        "diet_indication": "herbivore (browser)",
        "habitat_type": "terrestrial, desert and arid regions"
    },
    "cat": {
        "animal_class": "mammal",
        "distinctive_features": "retractable claws, whiskers, sharp teeth, flexible spine, vertical pupil slits, keen hearing",
        "body_coverage": "fur",
        "locomotion_type": "walking on four legs, excellent climber and jumper",
        "diet_indication": "carnivore (obligate carnivore)",
        "habitat_type": "terrestrial, domestic and semi-wild urban areas"
    },
    "chicken": {
        "animal_class": "bird",
        "distinctive_features": "red comb and wattles, beak, wings, tail feathers, spurs on legs, scratching feet",
        "body_coverage": "feathers",
        "locomotion_type": "walking on two legs, limited flying ability",
        "diet_indication": "omnivore (seeds, insects, small animals)",
        "habitat_type": "terrestrial, farms and domestic areas"
    },
    "dog": {
        "animal_class": "mammal",
        "distinctive_features": "non-retractable claws, wet nose, tail variations, floppy or erect ears, keen sense of smell",
        "body_coverage": "fur",
        "locomotion_type": "walking/running on four legs",
        "diet_indication": "omnivore with carnivore tendencies",
        "habitat_type": "terrestrial, domestic"
    },
    "dolphin": {
        "animal_class": "mammal",
        "distinctive_features": "streamlined body, dorsal fin, blow hole, flippers, echolocation, permanent smile expression",
        "body_coverage": "smooth skin (no fur)",
        "locomotion_type": "swimming with tail flukes",
        "diet_indication": "carnivore (piscivore - fish eater)",
        "habitat_type": "aquatic, oceans and some rivers"
    },
    "elephant": {
        "animal_class": "mammal",
        "distinctive_features": "trunk, large ears, tusks, thick wrinkled skin, pillar-like legs, largest land animal",
        "body_coverage": "thick skin with sparse hair",
        "locomotion_type": "walking on four legs",
        "diet_indication": "herbivore (browser and grazer)",
        "habitat_type": "terrestrial, savanna and forest"
    },
    "fox": {
        "animal_class": "mammal",
        "distinctive_features": "pointed ears, bushy tail, narrow snout, agile body, vertical slit pupils",
        "body_coverage": "thick fur",
        "locomotion_type": "walking/running on four legs",
        "diet_indication": "omnivore (small animals, berries, insects)",
        "habitat_type": "terrestrial, forests and grasslands"
    },
    "horse": {
        "animal_class": "mammal",
        "distinctive_features": "long face, flowing mane and tail, hooves, strong muscular body, large eyes on sides",
        "body_coverage": "short coat with long mane",
        "locomotion_type": "walking/running on four legs (hoofed)",
        "diet_indication": "herbivore (grazer)",
        "habitat_type": "terrestrial, grasslands and domestic"
    },
    "kangaroo": {
        "animal_class": "mammal (marsupial)",
        "distinctive_features": "powerful hind legs, long tail for balance, pouch for carrying young, small front legs",
        "body_coverage": "short fur",
        "locomotion_type": "hopping on hind legs",
        "diet_indication": "herbivore (grazer)",
        "habitat_type": "terrestrial, grasslands and open forests of Australia"
    },
    "koala": {
        "animal_class": "mammal (marsupial)",
        "distinctive_features": "large nose, fluffy ears, small eyes, strong claws for climbing, pouch for young",
        "body_coverage": "thick gray fur",
        "locomotion_type": "climbing, slow-moving",
        "diet_indication": "herbivore (eucalyptus leaves only)",
        "habitat_type": "arboreal, eucalyptus forests of Australia"
    },
    "polar_bear": {
        "animal_class": "mammal",
        "distinctive_features": "white fur, large paws for swimming, small ears, long neck, black skin underneath fur",
        "body_coverage": "thick white fur with dense undercoat",
        "locomotion_type": "walking on four legs, excellent swimmer",
        "diet_indication": "carnivore (mainly seals)",
        "habitat_type": "arctic regions, ice and cold coastal areas"
    },
    "red_panda": {
        "animal_class": "mammal",
        "distinctive_features": "reddish-brown fur, long bushy tail with rings, white face markings, semi-retractable claws",
        "body_coverage": "thick reddish fur",
        "locomotion_type": "climbing, walking on four legs",
        "diet_indication": "herbivore (primarily bamboo)",
        "habitat_type": "arboreal, mountain forests of Himalayas"
    },
    "seal": {
        "animal_class": "mammal",
        "distinctive_features": "streamlined body, flippers, whiskers, no external ears, layer of blubber",
        "body_coverage": "short fur over skin",
        "locomotion_type": "swimming with flippers, awkward on land",
        "diet_indication": "carnivore (fish and marine invertebrates)",
        "habitat_type": "amphibious, oceans and coastal areas"
    },
    "sheep": {
        "animal_class": "mammal",
        "distinctive_features": "woolly coat, rectangular pupils, split hooves, some have horns, docile nature",
        "body_coverage": "thick wool fleece",
        "locomotion_type": "walking on four legs",
        "diet_indication": "herbivore (grazer)",
        "habitat_type": "terrestrial, grasslands and farms"
    },
    "snow_leopard": {
        "animal_class": "mammal",
        "distinctive_features": "spotted fur pattern, long thick tail, large paws for snow walking, pale coat",
        "body_coverage": "thick spotted fur",
        "locomotion_type": "walking on four legs, excellent climber and jumper",
        "diet_indication": "carnivore (wild sheep, goats)",
        "habitat_type": "terrestrial, high mountain ranges of Central Asia"
    },
    "walrus": {
        "animal_class": "mammal",
        "distinctive_features": "long tusks, thick mustache whiskers, large body, wrinkled skin, flippers",
        "body_coverage": "thick skin with sparse hair",
        "locomotion_type": "swimming with flippers, slow on land",
        "diet_indication": "carnivore (mollusks and invertebrates)",
        "habitat_type": "amphibious, arctic coastal areas and ice"
    },
    "wombat": {
        "animal_class": "mammal (marsupial)",
        "distinctive_features": "stout body, short legs, large head, cube-shaped droppings, backward-facing pouch",
        "body_coverage": "coarse fur",
        "locomotion_type": "walking on four legs, excellent digger",
        "diet_indication": "herbivore (grasses and roots)",
        "habitat_type": "terrestrial and fossorial (burrowing), forests and grasslands of Australia"
    },
    "zebra": {
        "animal_class": "mammal",
        "distinctive_features": "distinctive black and white stripes, mane, hooves, horse-like body",
        "body_coverage": "short striped coat",
        "locomotion_type": "walking/running on four legs (hoofed)",
        "diet_indication": "herbivore (grazer)",
        "habitat_type": "terrestrial, grasslands and savannas of Africa"
    }
}

def gather_images(root):
    p = Path(root)
    files = [x for x in p.rglob("*") if x.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    return files

def infer_species(image_path):
    species_name = os.path.basename(os.path.dirname(str(image_path)))
    return species_name

def enrich_metadata_with_knowledge(species, visual_metadata):
    """
    Enrich visual metadata with species-specific biological knowledge
    """
    if species in SPECIES_KNOWLEDGE:
        knowledge = SPECIES_KNOWLEDGE[species]
        # Merge knowledge into metadata
        enriched = visual_metadata.copy()
        for key, value in knowledge.items():
            if key not in enriched or not enriched[key]:
                enriched[key] = value
        return enriched
    return visual_metadata

def index_folder(root_folder, weaviate_mode=True, dry_run=True, limit=None, detailed_metadata=False, model_name=None, model_key=None):

    # Determine model to use
    if model_key and model_key in settings.AVAILABLE_MODELS:
        model_info = settings.AVAILABLE_MODELS[model_key]
        model_name = model_info["model_id"]
        collection_name = model_info["collection"]
        logger.info(f"Using model: {model_info['name']} ({model_name})")
    else:
        collection_name = None
        logger.info(f"Using custom model: {model_name}")
    
    extractor = FeatureExtractor(model_name=model_name)
    files = gather_images(root_folder)
    if limit:
        files = files[:limit]
    logger.info("Found %d files", len(files))
    
    # Select questions based on detailed_metadata flag
    if detailed_metadata:
        questions = METADATA_QUESTIONS
        logger.info("Using DETAILED metadata extraction (11 visual fields + objects detection + species knowledge)")
    else:
        # Basic mode: essential visual features only
        questions = {
            "color_primary": METADATA_QUESTIONS["color_primary"],
            "action": METADATA_QUESTIONS["action"],
            "environment": METADATA_QUESTIONS["environment"],
            "distinctive_features": METADATA_QUESTIONS["distinctive_features"],
            "visible_objects": METADATA_QUESTIONS["visible_objects"],  # Include objects detection
        }
        logger.info("Using BASIC metadata extraction (5 visual fields with objects + species knowledge)")

    if weaviate_mode:
        client = get_weaviate_client()
        create_schema_if_not_exists(collection_name)
    else:
        client = None

    batch = []
    for p in tqdm(files, desc="Indexing"):
        pil = extractor.load_image(str(p))
        if pil is None:
            continue
        
        # Get species from folder name first
        folder_species = infer_species(p)
        
        # Extract visual metadata from image
        meta_answers = extractor.extract_metadata(pil, questions)
        
        # Enrich with species-specific biological knowledge
        meta_answers = enrich_metadata_with_knowledge(folder_species, meta_answers)
        
        props = {
            "file": str(p),
            "species": folder_species,
            "extra": json.dumps(meta_answers),
            "model_key": model_key or "custom"
        }
        vec = extractor.encode_image([pil])[0]
        vec = vec.tolist()
        # use seq id or a stable id
        
        batch.append({"properties": props, "vector": vec})


        if len(batch) >= settings.WEAVIATE_BATCH_SIZE:
            if not dry_run and weaviate_mode:
                batch_add_objects(client, batch, batch_size=settings.WEAVIATE_BATCH_SIZE, collection_name=collection_name)
            batch = []

    if batch:
        if not dry_run and weaviate_mode:
            batch_add_objects(client, batch, batch_size=settings.WEAVIATE_BATCH_SIZE, collection_name=collection_name)
    
    if client:
        client.close()
    
    logger.info("Indexing finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", required=True, help="Path to image folder")
    parser.add_argument("--weaviate", action="store_true", help="Use Weaviate for indexing")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually save to database")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--detailed-metadata", "--detailed", action="store_true", 
                        help="Extract detailed metadata (11 fields + objects, slower). Default: basic 5 fields")
    parser.add_argument("--model", type=str, default=None,
                        help="Custom CLIP model path (advanced users)")
    parser.add_argument("--model-key", type=str, choices=list(settings.AVAILABLE_MODELS.keys()),
                        help=f"Predefined model to use: {', '.join(settings.AVAILABLE_MODELS.keys())}. Each model uses separate collection.")
    args = parser.parse_args()
    index_folder(
        args.data_folder, 
        weaviate_mode=args.weaviate, 
        dry_run=args.dry_run, 
        limit=args.limit,
        detailed_metadata=args.detailed_metadata,
        model_name=args.model,
        model_key=args.model_key
    )
    # python -m app.indexer --data-folder data/full --weaviate --limit 100

