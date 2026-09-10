"""
Vector Store Module for RAG Pipeline.

Manages persistent ChromaDB vector storage and indexing operations.
Saves document chunks and their embedding vectors into the database.
"""

import logging
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

from app.config import CHROMA_DB_DIR, COLLECTION_NAME, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.ingestion import load_documents
from app.chunking import chunk_documents
from app.embeddings import embed_texts


# Suppress noisy telemetry logs
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

_CHROMA_SETTINGS = Settings(anonymized_telemetry=False)


def get_chroma_client():
    """Returns a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DB_DIR, settings=_CHROMA_SETTINGS)


def get_collection():
    """Retrieves or creates the target ChromaDB collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def build_index(data_dir: str = DATA_DIR, reset: bool = True, strategy: str = "recursive") -> int:
    """
    Wipes and rebuilds the collection from scratch using data_dir documents.

    Args:
        data_dir (str): Path to data directory.
        reset (bool): If True, deletes existing collection before building.
        strategy (str): 'recursive' or 'fixed' splitting.

    Returns:
        int: Number of text chunks indexed.
    """
    client = get_chroma_client()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # Collection didn't exist yet

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 1. Load documents
    docs = load_documents(data_dir)
    if not docs:
        print(f"No valid documents found in '{data_dir}'.")
        return 0

    # 2. Chunk documents (using recursive splitting strategy)
    chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, strategy=strategy)
    if not chunks:
        print("No chunks generated.")
        return 0

    # 3. Extract lists
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # 4. Generate embeddings & add to ChromaDB
    embeddings = embed_texts(texts)
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    return len(chunks)


if __name__ == "__main__":
    print("--- Testing Vector Store & Index Building ---")
    indexed_count = build_index()
    print(f"Indexed {indexed_count} chunks into ChromaDB collection '{COLLECTION_NAME}' at '{CHROMA_DB_DIR}'.")

