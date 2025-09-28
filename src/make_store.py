# src/make_store.py
import json, pickle
from pathlib import Path

DATA_DIR = Path("../data")
META_FILE = DATA_DIR / "emb_metadata.jsonl"
STORE_FILE = DATA_DIR / "store.pkl"

meta = []
with open(META_FILE, "r", encoding="utf-8") as f:
    for line in f:
        meta.append(json.loads(line))

with open(STORE_FILE, "wb") as f:
    pickle.dump(meta, f)

print("Wrote", STORE_FILE, "with", len(meta), "entries")
