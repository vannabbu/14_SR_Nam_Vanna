"""
Pipeline Module for RAG System.

Orchestrates Document Ingestion, Chunking, Vector Storage, Retrieval, and Generation.
"""

import os
import sys
from typing import Tuple, List, Dict, Any

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import DATA_DIR, TOP_K
from app.vector_store import build_index
from app.retriever import retrieve_context
from app.generator import generate_answer



def run_indexing_pipeline(data_dir: str = DATA_DIR, reset: bool = True) -> int:
    """
    Runs the offline ingestion, chunking, embedding, and vector store indexing setup.

    Returns:
        int: Number of document chunks indexed into ChromaDB.
    """
    print(f"Building vector index from '{data_dir}'...")
    count = build_index(data_dir=data_dir, reset=reset)
    print(f"Successfully indexed {count} text chunks into ChromaDB.")
    return count


def run_rag_pipeline(query: str, top_k: int = TOP_K) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Executes the online question-answering RAG pipeline.

    Args:
        query (str): User input question.
        top_k (int): Number of top context chunks to retrieve.

    Returns:
        Tuple[str, List[Dict[str, Any]]]: Synthesized answer and list of retrieved chunk dicts.
    """
    print(f"\nSearching vector store for: '{query}'...")
    retrieved_chunks = retrieve_context(query, top_k=top_k)

    print(f"Retrieved {len(retrieved_chunks)} relevant chunk(s). Synthesizing answer...")
    answer = generate_answer(query, retrieved_chunks)

    return answer, retrieved_chunks


class RAGPipeline:
    """Object-Oriented wrapper for RAG Pipeline."""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir

    def index_documents(self, reset: bool = True) -> int:
        return run_indexing_pipeline(data_dir=self.data_dir, reset=reset)

    def ask(self, query: str, top_k: int = TOP_K) -> Dict[str, Any]:
        answer, chunks = run_rag_pipeline(query, top_k=top_k)
        return {
            "query": query,
            "answer": answer,
            "sources": chunks
        }


if __name__ == "__main__":
    print("--- Testing RAG Pipeline ---")
    run_indexing_pipeline()
    ans, chunks = run_rag_pipeline("How do I set up my mobile device for company email?")
    print("\n--- Answer ---")
    print(ans)


