# src/assess_answer.py
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path
import faiss

DATA_DIR = Path("../data")
META_FILE = DATA_DIR / "emb_metadata.jsonl"
FAISS_INDEX_FILE = DATA_DIR / "faiss_index.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

def load_meta():
    meta = []
    with open(META_FILE, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta

def grade_answer(student_answer, reference_texts, weights=None):
    # Embed student and reference rubrics, compute average similarity
    all_texts = [student_answer] + reference_texts
    embs = model.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)
    s_emb = embs[0:1]
    refs = embs[1:]
    sims = cosine_similarity(s_emb, refs)[0]
    # aggregate
    score = float(sims.mean())  # 0..1
    # map to 0-100
    return round(score * 100, 1), sims.tolist()

def retrieve_top_k(query, k=5):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    D, I = index.search(q_emb, k)
    meta = load_meta()
    results = []
    for idx in I[0]:
        results.append(meta[idx])
    return results

if __name__ == "__main__":
    # example
    refs = [r["text"] for r in retrieve_top_k("What is inflation?", k=5)]
    s = "Inflation is rise in general price level due to increased money supply and demand."
    score, sims = grade_answer(s, refs)
    print("Score:", score, "sims:", sims)