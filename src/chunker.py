# src/chunker.py
import json, os, re
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("../data")
IN_JSONL = DATA_DIR / "chunks.jsonl"
OUT_CHUNKS = DATA_DIR / "emb_chunks.jsonl"

def simple_chunk(text, chunk_size=800, overlap=200):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk_tokens))
        i += (chunk_size - overlap)
    return chunks

def heuristic_topic(text):
    # Simple heuristics: look for lines that look like headings
    lines = text.splitlines()
    for ln in lines[:4]:
        ln_s = ln.strip()
        if len(ln_s) < 80 and len(ln_s.split()) > 1:
            return ln_s
    # fallback
    return None

def run():
    out = []
    with open(IN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks = simple_chunk(obj["text"], chunk_size=300, overlap=50)  # smaller for better retrieval
            topic = heuristic_topic(obj["text"])
            for i, c in enumerate(chunks):
                rec = {
                    "id": f"{obj['id']}_chunk_{i}",
                    "subject": obj.get("subject", "Economics"),
                    "topic": topic or "General",
                    "subtopic": None,
                    "difficulty": "medium",  # default; you can refine with heuristics
                    "page": obj["page"],
                    "text": c
                }
                out.append(rec)
    with open(OUT_CHUNKS, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Wrote", len(out), "chunks to", OUT_CHUNKS)

if __name__ == "__main__":
    run()
