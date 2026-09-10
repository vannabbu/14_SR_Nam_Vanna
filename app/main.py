"""
Main Interactive Chat Interface for RAG Pipeline.

Runs an interactive loop:
 1. Prompts the user for a question
 2. Calls the online RAG pipeline (retrieve -> generate)
 3. Prints the answer and retrieved context
 4. Repeats until the user types 'exit' or 'quit'
"""

import os
import sys

# Ensure package root is in sys.path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.pipeline import run_rag_pipeline, run_indexing_pipeline


def main():
    print("==================================================")
    print("        RAG Interactive Chat Assistant            ")
    print("==================================================")

    # Optional re-index check at startup
    if "--reindex" in sys.argv or "-r" in sys.argv:
        run_indexing_pipeline()

    print("\nSystem ready! Type your question below (or type 'exit' to quit).\n")

    while True:
        try:
            query = input("Question > ").strip()
            if not query:
                continue

            # Exit condition
            if query.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                break

            # Option to re-index on demand
            if query.lower() == "reindex":
                run_indexing_pipeline()
                continue

            # Execute RAG Pipeline (Retrieve -> Generate)
            answer, retrieved_chunks = run_rag_pipeline(query)

            # Print retrieved chunks for transparency
            print("\n[Retrieved Context]:")
            for idx, c in enumerate(retrieved_chunks, 1):
                source = c.get("source") or c.get("metadata", {}).get("source", "unknown")
                preview = c.get("text", "")[:120].replace("\n", " ")
                print(f"  {idx}. [{source}] {preview}...")

            # Print the synthesized LLM answer
            print(f"\nAnswer:\n{answer}\n")
            print("=" * 50 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()


