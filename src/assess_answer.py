# src/assess_answer.py
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path
import faiss
import os

DATA_DIR = Path("../data")
META_FILE = DATA_DIR / "emb_metadata.jsonl"
FAISS_INDEX_FILE = DATA_DIR / "faiss_index.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def load_meta():
    if not META_FILE.exists():
        print(f"[WARN] Metadata file {META_FILE} not found. Run build_index.py first.")
        return []
    meta = []
    with open(META_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                meta.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return meta


def grade_answer(student_answer, reference_texts, weights=None):
    if not student_answer.strip():
        return 0.0, []  # friendly return if empty answer
    if not reference_texts:
        return 0.0, []

    try:
        all_texts = [student_answer] + reference_texts
        embs = model.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)
        s_emb = embs[0:1]
        refs = embs[1:]
        sims = cosine_similarity(s_emb, refs)[0]
        score = float(sims.mean())  # 0..1
        return round(score * 100, 1), sims.tolist()
    except Exception as e:
        print(f"[ERROR] grading failed: {e}")
        return 0.0, []


def retrieve_top_k(query, k=5):
    if not FAISS_INDEX_FILE.exists():
        print(f"[WARN] FAISS index not found at {FAISS_INDEX_FILE}. Run build_index.py first.")
        return []
    if not query.strip():
        return []

    try:
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        index = faiss.read_index(str(FAISS_INDEX_FILE))
        D, I = index.search(q_emb, k)
        meta = load_meta()
        results = []
        for idx in I[0]:
            if 0 <= idx < len(meta):
                results.append(meta[idx])
        return results
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        return []


if __name__ == "__main__":
    refs = [r["text"] for r in retrieve_top_k("What is inflation?", k=5)]
    s = "Inflation is rise in general price level due to increased money supply and demand."
    score, sims = grade_answer(s, refs)
    print("Score:", score, "sims:", sims)
