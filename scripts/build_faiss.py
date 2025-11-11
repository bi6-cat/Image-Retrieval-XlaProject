import numpy as np, faiss, os
VEC_PATH = "data/vectors.npy"
IDX_PATH = os.getenv("FAISS_INDEX_PATH", "indices/faiss.index")
os.makedirs(os.path.dirname(IDX_PATH), exist_ok=True)
X = np.load(VEC_PATH).astype("float32")
print("Vectors shape:", X.shape)
faiss.normalize_L2(X)
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)
faiss.write_index(index, IDX_PATH)
print("Built FAISS index at", IDX_PATH, "ntotal=", index.ntotal)
