"""
ingest_pdf.py
-------------
Provides utility to load PDF files and extract their text content.
"""

from pathlib import Path
from PyPDF2 import PdfReader

def load_pdf(file_path: str) -> str:
    """
    Load a PDF file and return its full text.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Concatenated text from all pages of the PDF.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text
