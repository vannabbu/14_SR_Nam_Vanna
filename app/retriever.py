"""
Retriever Module for RAG Pipeline.

Queries ChromaDB vector collection to retrieve Top-K relevant context chunks.
Takes a question, embeds it using the embedding model, and returns matching context chunks.
"""

from typing import List, Dict, Any

from app.config import TOP_K
from app.vector_store import get_collection
from app.embeddings import embed_query


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Takes a question, embeds it into a vector, and queries ChromaDB for top matching chunks.

    Args:
        query (str): User question.
        top_k (int): Number of top context chunks to retrieve.

    Returns:
        List[Dict[str, Any]]: List of matching chunk dictionaries containing text, metadata, and distance.
    """
    collection = get_collection()
    query_vector = embed_query(query)

    if not query_vector:
        return []

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    retrieved_items = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
        ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

        for doc_text, meta, dist, chunk_id in zip(docs, metadatas, distances, ids):
            retrieved_items.append({
                "id": chunk_id,
                "text": doc_text,
                "metadata": meta,
                "source": meta.get("source", "unknown") if isinstance(meta, dict) else "unknown",
                "distance": dist
            })

    return retrieved_items


# Alias for backward compatibility
retrieve_context = retrieve


if __name__ == "__main__":
    print("--- Testing Retriever Module ---")
    test_query = "How do I set up company email on mobile?"
    matches = retrieve(test_query, top_k=2)
    print(f"Query: '{test_query}'\nFound {len(matches)} matches:")
    for m in matches:
        print(f"\n[Source: {m['source']}] Distance: {m['distance']:.4f}")
        print(f"Snippet: {m['text'][:150]}...")


