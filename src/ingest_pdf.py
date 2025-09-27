# src/ingest_pdf.py
import fitz  # PyMuPDF
import json
from pathlib import Path
from tqdm import tqdm

# Project root (go up from src/ to rag-assignment/)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

PDF_PATH = DATA_DIR / "chapter.pdf"
OUT_JSONL = DATA_DIR / "chunks.jsonl"

def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i + 1, "text": text})
    return pages

def save_chunks(pages, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pages:
            rec = {
                "id": f"page-{p['page']}",
                "subject": "Economics",
                "topic": None,      # we'll populate topic/subtopic later
                "subtopic": None,
                "difficulty": None,
                "page": p['page'],
                "text": p['text'].strip()
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Saved:", out_path)

if __name__ == "__main__":
    pages = extract_pages(PDF_PATH)
    save_chunks(pages, OUT_JSONL)
