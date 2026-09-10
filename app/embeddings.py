"""
Embeddings Module for RAG Pipeline.

Turns text chunks and user queries into dense vector representations.
Uses local Ollama (`nomic-embed-text`) with fallback to ChromaDB's default embedding function.
"""

from typing import List

from app.config import EMBED_MODEL


import ollama
from chromadb.utils import embedding_functions

_fallback_fn = embedding_functions.DefaultEmbeddingFunction()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generates numerical vector embeddings for a list of text strings.

    Args:
        texts (List[str]): List of text strings to embed.

    Returns:
        List[List[float]]: Matrix of float vector embeddings.
    """
    if not texts:
        return []

    try:
        response = ollama.embed(model=EMBED_MODEL, input=texts)
        return list(response.embeddings)  # type: ignore
    except Exception as e:
        print(f"[Warning] Ollama embedding with '{EMBED_MODEL}' failed ({e}). Using default ONNX fallback.")
        return _fallback_fn(texts)


def embed_query(query: str) -> List[float]:
    """
    Convenience wrapper to embed a single query string.

    Args:
        query (str): User question/query.

    Returns:
        List[float]: Single vector embedding.
    """
    if not query.strip():
        return []
    res = embed_texts([query])
    return res[0] if res else []


if __name__ == "__main__":
    print("--- Testing Embeddings Module ---")
    sample_texts = ["How do I set up my company email on mobile?", "VPN configuration steps."]
    vectors = embed_texts(sample_texts)
    print(f"Generated {len(vectors)} vectors.")
    if vectors:
        print(f"Vector dimension size: {len(vectors[0])}")

