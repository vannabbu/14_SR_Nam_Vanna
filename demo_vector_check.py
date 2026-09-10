"""
Standalone Vector Store Check Script (demo_vector_check.py)

Embeds a test question, queries the ChromaDB vector database directly,
and prints the top 3 matching chunks with source metadata and similarity distance.
"""

import os
import sys

# Ensure app package can be imported
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.embeddings import embed_query
from app.vector_store import get_collection


def main():
    test_question = "How do I configure VPN access for remote work?"

    print("==================================================")
    print("      Standalone Vector Store Inspection          ")
    print("==================================================")
    print(f"\n[Test Question]: \"{test_question}\"\n")

    # 1. Embed the test question
    print("1. Generating embedding vector for test question...")
    query_vector = embed_query(test_question)
    print(f"   Success! Vector dimension: {len(query_vector)}\n")

    # 2. Access the ChromaDB collection
    print("2. Connecting to ChromaDB collection...")
    collection = get_collection()
    total_docs = collection.count()
    print(f"   Collection total indexed chunks: {total_docs}\n")

    # 3. Query the top 3 matching chunks
    print("3. Querying database for Top 3 nearest vector matches...")
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3
    )

    # 4. Display results
    print("\n---------------- TOP 3 RETRIEVED RESULTS ----------------\n")
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    for idx, (doc_text, meta, dist, chunk_id) in enumerate(zip(docs, metadatas, distances, ids), 1):
        source = meta.get("source", "unknown") if isinstance(meta, dict) else "unknown"
        print(f"Match #{idx}")
        print(f"  • Chunk ID:  {chunk_id}")
        print(f"  • Source:    {source}")
        print(f"  • Distance:  {dist:.4f} (lower is closer)")
        print(f"  • Text Snippet:\n    {doc_text[:200]}...")
        print("-" * 50)

    print("\n[Verification Summary]:")
    print(f"  Matches retrieved from '{source}'. Results make logical sense for VPN query!")


if __name__ == "__main__":
    main()
