#!/usr/bin/env python3
"""
Performance Testing for Image Retrieval System
Measures encoding and search times for different operations
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.encoder import Encoder
from app.weaviate_client import get_weaviate_client
from PIL import Image


def test_text_encoding_speed(model_name="openai/clip-vit-base-patch32", num_samples=100):
    """Test text encoding speed"""
    print(f"\n=== Testing Text Encoding Speed ({model_name}) ===")
    
    encoder = Encoder(model_name=model_name)
    
    test_queries = [
        "cute cat sleeping",
        "dog running on beach",
        "bird flying in sky",
        "elephant in nature",
        "butterfly on flower",
        "horse running in field",
        "cow grazing in meadow",
        "sheep on hillside"
    ]
    
    times = []
    for i in range(num_samples):
        query = test_queries[i % len(test_queries)]
        
        start = time.time()
        embedding = encoder.encode_text([query])
        elapsed = time.time() - start
        
        times.append(elapsed * 1000)  # Convert to milliseconds
    
    print(f"Samples: {num_samples}")
    print(f"Average: {np.mean(times):.2f}ms")
    print(f"Min: {np.min(times):.2f}ms")
    print(f"Max: {np.max(times):.2f}ms")
    print(f"Std Dev: {np.std(times):.2f}ms")
    
    return {
        "operation": "text_encoding",
        "model": model_name,
        "samples": num_samples,
        "avg_ms": np.mean(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "std_ms": np.std(times)
    }


def test_image_encoding_speed(model_name="openai/clip-vit-base-patch32", num_samples=50):
    """Test image encoding speed"""
    print(f"\n=== Testing Image Encoding Speed ({model_name}) ===")
    
    encoder = Encoder(model_name=model_name)
    
    # Find test images
    data_dir = Path("data/full")
    test_images = []
    
    for species_dir in data_dir.iterdir():
        if species_dir.is_dir():
            images = list(species_dir.glob("*.jpg"))[:5]  # 5 images per species
            test_images.extend(images)
            if len(test_images) >= num_samples:
                break
    
    test_images = test_images[:num_samples]
    
    if len(test_images) == 0:
        print("ERROR: No test images found!")
        return None
    
    times = []
    for img_path in test_images:
        try:
            img = Image.open(img_path).convert('RGB')
            
            start = time.time()
            embedding = encoder.encode_images([img])
            elapsed = time.time() - start
            
            times.append(elapsed * 1000)  # Convert to milliseconds
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Samples: {len(times)}")
    print(f"Average: {np.mean(times):.2f}ms")
    print(f"Min: {np.min(times):.2f}ms")
    print(f"Max: {np.max(times):.2f}ms")
    print(f"Std Dev: {np.std(times):.2f}ms")
    
    return {
        "operation": "image_encoding",
        "model": model_name,
        "samples": len(times),
        "avg_ms": np.mean(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "std_ms": np.std(times)
    }


def test_search_speed(collection_name="AnimalImageClipBase", num_samples=100):
    """Test Weaviate search speed"""
    print(f"\n=== Testing Search Speed ({collection_name}) ===")
    
    client = get_weaviate_client()
    encoder = Encoder(model_name="openai/clip-vit-base-patch32")
    
    test_queries = [
        "cute cat", "brown dog", "flying bird", "large elephant",
        "colorful butterfly", "running horse", "white sheep", "black cow"
    ]
    
    times = []
    for i in range(num_samples):
        query = test_queries[i % len(test_queries)]
        
        # Encode query
        query_vector = encoder.encode_text([query])[0]
        
        # Time the search only
        start = time.time()
        collection = client.collections.get(collection_name)
        response = collection.query.near_vector(
            near_vector=query_vector.tolist(),
            limit=20
        )
        elapsed = time.time() - start
        
        times.append(elapsed * 1000)  # Convert to milliseconds
    
    client.close()
    
    print(f"Samples: {num_samples}")
    print(f"Average: {np.mean(times):.2f}ms")
    print(f"Min: {np.min(times):.2f}ms")
    print(f"Max: {np.max(times):.2f}ms")
    print(f"Std Dev: {np.std(times):.2f}ms")
    
    return {
        "operation": "weaviate_search",
        "collection": collection_name,
        "samples": num_samples,
        "avg_ms": np.mean(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "std_ms": np.std(times)
    }


def test_end_to_end(num_samples=50):
    """Test complete end-to-end text search"""
    print(f"\n=== Testing End-to-End Text Search ===")
    
    client = get_weaviate_client()
    encoder = Encoder(model_name="openai/clip-vit-base-patch32")
    
    test_queries = [
        "cute cat sleeping on sofa",
        "dog running on beach at sunset",
        "bird flying in blue sky",
        "elephant walking in savanna",
        "colorful butterfly on flower"
    ]
    
    total_times = []
    encode_times = []
    search_times = []
    
    for i in range(num_samples):
        query = test_queries[i % len(test_queries)]
        
        # Total time
        start_total = time.time()
        
        # Encode time
        start_encode = time.time()
        query_vector = encoder.encode_text([query])[0]
        encode_time = time.time() - start_encode
        
        # Search time
        start_search = time.time()
        collection = client.collections.get("AnimalImageClipBase")
        response = collection.query.near_vector(
            near_vector=query_vector.tolist(),
            limit=20
        )
        search_time = time.time() - start_search
        
        total_time = time.time() - start_total
        
        total_times.append(total_time * 1000)
        encode_times.append(encode_time * 1000)
        search_times.append(search_time * 1000)
    
    client.close()
    
    print(f"Samples: {num_samples}")
    print(f"Total Average: {np.mean(total_times):.2f}ms")
    print(f"  - Encode: {np.mean(encode_times):.2f}ms")
    print(f"  - Search: {np.mean(search_times):.2f}ms")
    
    return {
        "operation": "end_to_end_text_search",
        "samples": num_samples,
        "total_avg_ms": np.mean(total_times),
        "encode_avg_ms": np.mean(encode_times),
        "search_avg_ms": np.mean(search_times)
    }


def compare_models():
    """Compare ViT-B/32 vs ViT-B/16"""
    print("\n" + "="*60)
    print("COMPARING MODELS: ViT-B/32 vs ViT-B/16")
    print("="*60)
    
    results = []
    
    # Test ViT-B/32
    print("\n--- Model: CLIP ViT-B/32 ---")
    results.append(test_text_encoding_speed("openai/clip-vit-base-patch32", num_samples=50))
    results.append(test_image_encoding_speed("openai/clip-vit-base-patch32", num_samples=30))
    
    # Test ViT-B/16
    print("\n--- Model: CLIP ViT-B/16 ---")
    results.append(test_text_encoding_speed("openai/clip-vit-base-patch16", num_samples=50))
    results.append(test_image_encoding_speed("openai/clip-vit-base-patch16", num_samples=30))
    
    # Summary comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    for i in range(0, len(results), 2):
        b32_text = results[i]
        b32_image = results[i+1]
        
        if i+2 < len(results):
            b16_text = results[i+2]
            b16_image = results[i+3]
            
            print("\nText Encoding:")
            print(f"  ViT-B/32: {b32_text['avg_ms']:.2f}ms")
            print(f"  ViT-B/16: {b16_text['avg_ms']:.2f}ms")
            print(f"  Speedup: {b16_text['avg_ms']/b32_text['avg_ms']:.2f}x slower")
            
            print("\nImage Encoding:")
            print(f"  ViT-B/32: {b32_image['avg_ms']:.2f}ms")
            print(f"  ViT-B/16: {b16_image['avg_ms']:.2f}ms")
            print(f"  Speedup: {b16_image['avg_ms']/b32_image['avg_ms']:.2f}x slower")


def run_all_tests():
    """Run all performance tests"""
    print("\n" + "="*60)
    print("IMAGE RETRIEVAL SYSTEM - PERFORMANCE TESTING")
    print("="*60)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    
    results = []
    
    # Individual tests
    results.append(test_text_encoding_speed(num_samples=100))
    results.append(test_image_encoding_speed(num_samples=50))
    results.append(test_search_speed(num_samples=100))
    results.append(test_end_to_end(num_samples=50))
    
    # Model comparison
    compare_models()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
    
    return results


if __name__ == "__main__":
    try:
        results = run_all_tests()
        
        # Save results to file
        output_file = "test_results_performance.txt"
        with open(output_file, "w") as f:
            f.write("Performance Test Results\n")
            f.write("="*60 + "\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for result in results:
                f.write(f"Operation: {result['operation']}\n")
                for key, value in result.items():
                    if key != 'operation':
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
