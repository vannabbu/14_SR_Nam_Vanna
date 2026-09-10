"""
Text Chunking Module for RAG Pipeline.

Implements text splitting strategies to break raw document text into smaller chunks.
Strategies provided:
 1. Fixed-size Splitting with Overlap (character-based sliding window)
 2. Recursive Character Splitting (tries splitting by paragraph '\\n\\n', line '\\n', sentence '. ', space ' ', then characters)

Configuration:
 - Chunk Size: 400 characters (within 300–500 target range)
 - Chunk Overlap: 50 characters (12.5% overlap, fitting the 10%–20% target range)
"""

from typing import List, Dict, Any


DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 50  # 12.5% overlap to preserve context across boundaries


def chunk_text_fixed_size(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """
    Fixed-size character splitting with overlap.

    Args:
        text (str): Input text content.
        chunk_size (int): Max characters per chunk (default: 400).
        chunk_overlap (int): Number of overlapping characters between adjacent chunks (default: 50).

    Returns:
        List[str]: List of text chunk strings.
    """
    text = text.strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = end - chunk_overlap  # step back to share context with the next chunk

    return chunks


def chunk_text_recursive(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: List[str] = None
) -> List[str]:
    """
    Recursive character splitting. Recursively splits text by paragraph, line, 
    sentence, or space to keep semantically related sentences together.

    Args:
        text (str): Input text content.
        chunk_size (int): Target max characters per chunk.
        chunk_overlap (int): Overlap between adjacent chunks.
        separators (List[str], optional): Hierarchical separators.

    Returns:
        List[str]: List of semantically split text chunk strings.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    text = text.strip()
    if not text:
        return []

    # If text is already within chunk size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]

    # Find the best separator to use
    separator = separators[-1]
    for sep in separators:
        if sep == "" or sep in text:
            separator = sep
            break

    # Split by the chosen separator
    if separator != "":
        splits = text.split(separator)
    else:
        splits = list(text)

    # Recombine splits up to chunk_size with overlap
    final_chunks: List[str] = []
    current_doc: List[str] = []
    current_length = 0

    for s in splits:
        piece = s if separator == "" else s + separator
        piece_len = len(piece)

        if current_length + piece_len > chunk_size and current_doc:
            joined_chunk = "".join(current_doc).strip()
            if joined_chunk:
                final_chunks.append(joined_chunk)
            
            # Keep overlap context from the end of current_doc
            overlap_buffer: List[str] = []
            overlap_len = 0
            for prev_item in reversed(current_doc):
                if overlap_len + len(prev_item) <= chunk_overlap:
                    overlap_buffer.insert(0, prev_item)
                    overlap_len += len(prev_item)
                else:
                    break
            current_doc = overlap_buffer
            current_length = overlap_len

        current_doc.append(piece)
        current_length += piece_len

    if current_doc:
        joined_chunk = "".join(current_doc).strip()
        if joined_chunk:
            final_chunks.append(joined_chunk)

    return final_chunks


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    strategy: str = "fixed"
) -> List[Dict[str, Any]]:
    """
    Chunks a collection of document dicts (containing 'filename' and 'text').

    Args:
        documents (List[Dict]): List of dicts e.g. [{"filename": "doc.txt", "text": "..."}]
        chunk_size (int): Size limit per chunk.
        chunk_overlap (int): Overlap size.
        strategy (str): 'fixed' or 'recursive'.

    Returns:
        List[Dict]: Chunk objects with id, text, and metadata.
    """
    processed_chunks = []

    for doc in documents:
        filename = doc.get("filename", "unknown")
        text = doc.get("text", "")

        if strategy == "recursive":
            raw_chunks = chunk_text_recursive(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            raw_chunks = chunk_text_fixed_size(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for i, chunk in enumerate(raw_chunks):
            processed_chunks.append({
                "id": f"{filename}::{i}",
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                    "chunk_size": len(chunk),
                }
            })

    return processed_chunks


if __name__ == "__main__":
    print("--- Testing chunking.py ---")
    sample_text = (
        "Travel and Expense Policy — Novalink Devices\n\n"
        "Approval\n"
        "Any business trip expected to cost more than $500 requires manager approval before booking. "
        "Trips under $500 (e.g. a local client visit) can be booked directly and expensed afterward.\n\n"
        "Booking\n"
        "Flights and hotels should be booked through the company travel portal whenever possible. "
        "Economy class is standard for flights under 6 hours; premium economy is allowed for flights over 6 hours.\n\n"
        "Meals\n"
        "Meal expenses while traveling are reimbursed up to $60/day, itemized where possible."
    )

    print(f"Sample text total length: {len(sample_text)} characters\n")

    # 1. Fixed-size Splitting
    fixed_chunks = chunk_text_fixed_size(sample_text, chunk_size=300, chunk_overlap=50)
    print(f"Fixed-size Splitting (chunk_size=300, overlap=50) -> {len(fixed_chunks)} chunks:")
    for idx, c in enumerate(fixed_chunks):
        print(f"  Chunk [{idx}] ({len(c)} chars): {c[:70]}...")

    print("\n" + "="*50 + "\n")

    # 2. Recursive Splitting
    recursive_chunks = chunk_text_recursive(sample_text, chunk_size=300, chunk_overlap=50)
    print(f"Recursive Character Splitting (chunk_size=300, overlap=50) -> {len(recursive_chunks)} chunks:")
    for idx, c in enumerate(recursive_chunks):
        print(f"  Chunk [{idx}] ({len(c)} chars): {c[:70]}...")
