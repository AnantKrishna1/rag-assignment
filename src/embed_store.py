# src/embed_store.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

# Project root aware paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Input: chunks produced by ingest_pdf.py
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"

# Outputs
EMB_OUT = DATA_DIR / "embeddings.npy"
META_OUT = DATA_DIR / "emb_metadata.jsonl"
FAISS_INDEX_FILE = DATA_DIR / "faiss_index.index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_chunks(path):
    if not path.exists():
        print(f"❌ Chunks file not found: {path}")
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def build_index(chunks):
    if not chunks:
        print("⚠️ No chunks to embed! Did you run ingest_pdf.py?")
        return
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    print("Encoding", len(texts), "chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors -> inner product
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    # Save embeddings and metadata
    np.save(EMB_OUT, embeddings)
    with open(META_OUT, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("✅ Saved FAISS index and metadata")

def main():
    chunks = load_chunks(CHUNKS_FILE)
    build_index(chunks)

if __name__ == "__main__":
    main()
