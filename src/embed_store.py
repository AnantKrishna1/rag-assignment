# src/embed_store.py
"""
Embed Store Module
-----------------
This module handles the creation of embeddings from PDF/text chunks
and builds a FAISS index for semantic search.

Functions:
- load_chunks(path): Load preprocessed text chunks from JSONL file.
- build_index(chunks): Encode chunks, save embeddings, metadata, and FAISS index.
- build_embed_store(): Wrapper to load chunks and build the full embed store.
"""

import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Project root-aware paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Input chunks file
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"

# Output files
EMB_OUT = DATA_DIR / "embeddings.npy"
META_OUT = DATA_DIR / "emb_metadata.jsonl"
FAISS_INDEX_FILE = DATA_DIR / "faiss_index.index"

# SentenceTransformer model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: Path) -> list[dict]:
    """
    Load text chunks from a JSONL file.

    Args:
        path (Path): Path to the JSONL file containing chunks.

    Returns:
        List[dict]: List of chunk dictionaries.
    """
    if not path.exists():
        print(f"❌ Chunks file not found: {path}")
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_index(chunks: list[dict]) -> None:
    """
    Build embeddings and a FAISS index from chunks and save them to disk.

    Args:
        chunks (List[dict]): List of chunk dictionaries with 'text' keys.
    """
    if not chunks:
        print("⚠️ No chunks to embed! Did you run ingest_pdf.py?")
        return

    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    print(f"Encoding {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # FAISS index (cosine similarity via normalized vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_FILE))

    # Save embeddings
    np.save(EMB_OUT, embeddings)

    # Save metadata
    with open(META_OUT, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("✅ FAISS index, embeddings, and metadata saved.")


def build_embed_store() -> None:
    """
    Load chunks and build the full embed store.
    This is the main recruiter-facing entry point.
    """
    chunks = load_chunks(CHUNKS_FILE)
    build_index(chunks)


if __name__ == "__main__":
    build_embed_store()
