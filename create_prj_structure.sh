#!/bin/bash

# Thư mục gốc
ROOT_DIR="image-retrieval"

# Tạo thư mục gốc và chuyển vào đó
mkdir -p $ROOT_DIR
cd $ROOT_DIR

# 1. Tạo thư mục 'app' và các tệp bên trong
mkdir -p app
touch app/__init__.py app/config.py app/utils.py app/encoder.py app/extractor.py app/weaviate_client.py app/indexer.py app/api.py

# 2. Tạo thư mục 'frontend' và các tệp bên trong
mkdir -p frontend
touch frontend/index.html frontend/script.js frontend/styles.css

# 3. Tạo thư mục 'scripts' và các tệp bên trong
mkdir -p scripts
touch scripts/generate_demo_data.py scripts/build_faiss.py

# 4. Tạo các tệp ở thư mục gốc
touch docker-compose.yml .env.example requirements.txt README_RUN.md

# Quay lại thư mục ban đầu (Tùy chọn)
cd ..

echo "Tạo cấu trúc thư mục '$ROOT_DIR' thành công!"