import os, json, numpy as np
os.makedirs("data", exist_ok=True)
N = 200
D = int(os.getenv("VECTOR_DIM", "512"))
rng = np.random.default_rng(42)
X = rng.standard_normal((N, D)).astype("float32")
np.save("data/vectors.npy", X)
ids = np.arange(N, dtype=np.int32)
np.save("data/ids.npy", ids)
meta = {}
species = ["cat","dog","bird","horse","tiger","lion"]
for i in range(N):
    meta[str(i)] = {
        "file": f"images/{species[i % len(species)]}_{i:03d}.jpg",
        "caption": f"{species[i % len(species)].capitalize()} sample {i}",
        "species": species[i % len(species)]
    }
with open("data/id2meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("Demo data created.")
