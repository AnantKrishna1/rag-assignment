# src/test_retrieval.py
from sentence_transformers import SentenceTransformer
import faiss, json
from pathlib import Path

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL)
INDEX = Path("../data/faiss_index.index")
META = Path("../data/emb_metadata.jsonl")

index = faiss.read_index(str(INDEX))
with open(META, "r", encoding="utf-8") as f:
    meta = [json.loads(l) for l in f]

q = "What is inflation?"
q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
D, I = index.search(q_emb, 5)
print("Top results (distance scores):")
for dist, idx in zip(D[0], I[0]):
    if idx < 0: continue
    print(dist, meta[idx]["page"], meta[idx]["topic"])
    print(meta[idx]["text"][:300].replace("\n"," "), "...")
    print("----")
