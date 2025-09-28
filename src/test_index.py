# src/test_index.py
import faiss, json
from pathlib import Path

INDEX = Path("../data/faiss_index.index")
META = Path("../data/emb_metadata.jsonl")

idx = faiss.read_index(str(INDEX))
with open(META, "r", encoding="utf-8") as f:
    meta = [json.loads(l) for l in f]

print("FAISS ntotal:", idx.ntotal)
print("Metadata count:", len(meta))
assert idx.ntotal == len(meta), "Index and metadata length mismatch!"
print("Index and metadata sizes match ✅")
