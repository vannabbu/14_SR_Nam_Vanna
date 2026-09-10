"""
Document Ingestion Module for RAG Pipeline.

Loads all supported documents (.txt, .md, .pdf) from the specified data directory.
"""

import os
from typing import List, Dict, Any

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    PdfReader = None

from app.config import DATA_DIR


def _read_txt(path: str) -> str:
    """Reads a plain text or markdown file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    """Extracts text from a PDF file using pypdf."""
    if not HAS_PYPDF or PdfReader is None:
        print(f"[Warning] 'pypdf' library not found. Skipping PDF: {path}")
        return ""
    reader = PdfReader(path)
    text_pages = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text_pages.append(extracted)
    return "\n".join(text_pages)



def load_documents(data_dir: str = DATA_DIR) -> List[Dict[str, Any]]:
    """
    Scans data_dir and loads text content from all .txt, .md, and .pdf files.

    Args:
        data_dir (str): Path to directory containing source documents.

    Returns:
        List[Dict[str, Any]]: List of document dictionaries with 'filename', 'text', and 'metadata'.
    """
    if not os.path.exists(data_dir):
        print(f"Warning: Directory '{data_dir}' does not exist.")
        return []

    documents = []
    for filename in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            continue

        ext = filename.lower().rsplit(".", 1)[-1]
        text = ""

        try:
            if ext in ("txt", "md"):
                text = _read_txt(path)
            elif ext == "pdf":
                text = _read_pdf(path)
            else:
                continue  # Skip unsupported file extensions

        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            continue

        if text.strip():
            documents.append({
                "filename": filename,
                "text": text,
                "metadata": {
                    "source": filename,
                    "file_path": path,
                    "file_type": ext,
                    "char_count": len(text)
                }
            })

    return documents


if __name__ == "__main__":
    print("--- Testing Ingestion Module ---")
    docs = load_documents()
    print(f"Loaded {len(docs)} document(s) from '{DATA_DIR}':")
    for doc in docs:
        print(f" - {doc['filename']} ({doc['metadata']['char_count']} chars)")

