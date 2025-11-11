#!/bin/bash
# Script to download animal image datasets

echo "🐾 Animal Image Dataset Downloader"
echo "=================================="
echo ""

# Create data directory
mkdir -p data/downloads
cd data/downloads

echo "📦 Available Datasets:"
echo ""
echo "1. Animals-10 (Small - 28K images, ~3GB)"
echo "   10 categories: dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, elephant"
echo "   Source: Kaggle"
echo ""
echo "2. ImageNet Animals Subset (Medium - 100K images, ~15GB)"
echo "   100+ animal categories with detailed labels"
echo "   Source: ImageNet"
echo ""
echo "3. Open Images Animals (Large - 500K+ images, ~80GB)"
echo "   600+ animal categories with bounding boxes"
echo "   Source: Google Open Images"
echo ""
echo "4. iNaturalist 2021 (Very Large - 2.7M images, ~300GB)"
echo "   10,000 species of animals, plants, fungi"
echo "   Source: iNaturalist"
echo ""

# Function to download Animals-10
download_animals10() {
    echo "📥 Downloading Animals-10 dataset..."
    echo ""
    echo "⚠️  This requires Kaggle API credentials!"
    echo "Setup instructions:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Create API token (Downloads kaggle.json)"
    echo "3. Place at ~/.kaggle/kaggle.json"
    echo "4. chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    read -p "Have you set up Kaggle API? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Install kaggle
        pip install -q kaggle
        
        # Download dataset
        kaggle datasets download -d alessiocorrado99/animals10
        
        # Extract
        echo "📦 Extracting..."
        unzip -q animals10.zip -d animals10
        rm animals10.zip
        
        echo "✅ Downloaded to: data/downloads/animals10/"
        echo "📊 Statistics:"
        find animals10 -type f | wc -l | xargs echo "   Total images:"
        du -sh animals10 | awk '{print "   Size: " $1}'
        
        echo ""
        echo "🚀 To index this dataset:"
        echo "python app/indexer.py --data-folder data/downloads/animals10 --weaviate"
    else
        echo "❌ Skipped. Please set up Kaggle API first."
    fi
}

# Function to download sample from Unsplash
download_sample() {
    echo "📥 Downloading curated sample dataset (100 images)..."
    echo "Source: Unsplash (free, no API needed)"
    echo ""
    
    mkdir -p sample
    cd sample
    
    # Animal keywords
    animals=("cat" "dog" "elephant" "lion" "tiger" "bird" "fish" "horse" "cow" "sheep")
    
    for animal in "${animals[@]}"; do
        mkdir -p "$animal"
        echo "Downloading ${animal} images..."
        
        for i in {1..10}; do
            # Using Unsplash Source API (random images)
            curl -L -o "${animal}/img_${i}.jpg" \
                "https://source.unsplash.com/800x600/?${animal},animal" \
                2>/dev/null
            sleep 1  # Rate limiting
        done
    done
    
    cd ..
    
    echo "✅ Downloaded to: data/downloads/sample/"
    echo "📊 100 images across 10 categories"
    echo ""
    echo "🚀 To index this dataset:"
    echo "python app/indexer.py --data-folder data/downloads/sample --weaviate"
}

# Function for manual download instructions
manual_download() {
    echo "📖 Manual Download Instructions"
    echo "==============================="
    echo ""
    echo "🔗 Dataset Sources:"
    echo ""
    echo "1. COCO Animals Subset"
    echo "   URL: https://cocodataset.org/"
    echo "   Size: ~20GB"
    echo "   Categories: 80+ including many animals"
    echo ""
    echo "2. ImageNet"
    echo "   URL: https://www.image-net.org/"
    echo "   Size: Varies (can select specific synsets)"
    echo "   Note: Requires account registration"
    echo ""
    echo "3. Oxford-IIIT Pet Dataset"
    echo "   URL: https://www.robots.ox.ac.uk/~vgg/data/pets/"
    echo "   Size: ~800MB"
    echo "   Categories: 37 cat and dog breeds"
    echo "   wget: https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    echo ""
    echo "4. Stanford Dogs Dataset"
    echo "   URL: http://vision.stanford.edu/aditya86/ImageNetDogs/"
    echo "   Size: ~750MB"
    echo "   Categories: 120 dog breeds"
    echo ""
    echo "5. Caltech-UCSD Birds-200-2011"
    echo "   URL: http://www.vision.caltech.edu/datasets/cub_200_2011/"
    echo "   Size: ~1.2GB"
    echo "   Categories: 200 bird species"
    echo ""
    echo "6. iWildCam - Wildlife Camera Traps"
    echo "   URL: https://github.com/visipedia/iwildcam_comp"
    echo "   Size: ~100GB"
    echo "   Categories: Wildlife in natural habitats"
    echo ""
}

# Quick download function for Oxford Pets (easy, no API needed)
download_oxford_pets() {
    echo "📥 Downloading Oxford-IIIT Pet Dataset..."
    echo "Size: ~800MB, 37 breeds of cats and dogs"
    echo ""
    
    mkdir -p oxford_pets
    cd oxford_pets
    
    echo "Downloading images..."
    wget -q --show-progress https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    
    echo "Extracting..."
    tar -xzf images.tar.gz
    rm images.tar.gz
    
    # Organize by breed (folder name is breed)
    echo "Organizing..."
    mkdir -p organized
    for img in images/*.jpg; do
        # Get breed name (everything before last underscore and number)
        breed=$(basename "$img" | sed 's/_[0-9]*\.jpg$//' | tr '[:upper:]' '[:lower:]')
        mkdir -p "organized/$breed"
        cp "$img" "organized/$breed/"
    done
    
    cd ..
    
    echo "✅ Downloaded to: data/downloads/oxford_pets/organized/"
    find oxford_pets/organized -type f | wc -l | xargs echo "📊 Total images:"
    du -sh oxford_pets | awk '{print "   Size: " $1}'
    echo ""
    echo "🚀 To index this dataset:"
    echo "python app/indexer.py --data-folder data/downloads/oxford_pets/organized --weaviate --detailed"
}

# Main menu
echo "Choose an option:"
echo "1. Download sample dataset (100 images, ~10MB) - QUICK"
echo "2. Download Oxford Pets (7,400 images, ~800MB) - RECOMMENDED"
echo "3. Download Animals-10 via Kaggle (28K images, ~3GB)"
echo "4. Show manual download instructions"
echo "5. Exit"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        download_sample
        ;;
    2)
        download_oxford_pets
        ;;
    3)
        download_animals10
        ;;
    4)
        manual_download
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✨ Done! Next steps:"
echo "1. Index the dataset with indexer.py"
echo "2. Start the API server"
echo "3. Search and enjoy!"
echo ""
echo "📚 For detailed metadata (slower but better):"
echo "   Add --detailed flag when indexing"
echo ""
echo "Example:"
echo "python app/indexer.py --data-folder data/downloads/oxford_pets/organized --weaviate --detailed --limit 100"
