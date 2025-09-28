"""
build_index.py
--------------
Builds a FAISS vector index from PDFs in the data directory.
"""

from pathlib import Path
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from ingest_pdf import load_pdf

# =====================
# Configuration
# =====================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = DATA_DIR / "faiss_index.index"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"

# =====================
# Load Model
# =====================
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# =====================
# Collect PDFs
# =====================
pdf_files = list(DATA_DIR.glob("*.pdf"))
if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in {DATA_DIR}")

# =====================
# Extract Text
# =====================
documents = []
for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
    text = load_pdf(pdf_file)
    documents.append(text)

# =====================
# Compute Embeddings
# =====================
print("Computing embeddings...")
embeddings = model.encode(documents, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Save embeddings for future use
np.save(EMBEDDINGS_PATH, embeddings)

# =====================
# Build FAISS Index
# =====================
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, str(INDEX_PATH))

print(f"FAISS index built successfully with {len(documents)} documents.")
print(f"Index saved to: {INDEX_PATH}")
print(f"Embeddings saved to: {EMBEDDINGS_PATH}")
