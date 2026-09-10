"""
Generator Module for RAG Pipeline.

Synthesizes a grounded answer to the user's question using the retrieved context chunks.
Sends structured prompt with retrieved context to local Ollama LLM.
"""

from typing import List, Dict, Any
import ollama

from app.config import GEN_MODEL, SYSTEM_PROMPT



def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks and user question into a structured LLM prompt."""
    if not chunks:
        context_block = "(no relevant context was found)"
    else:
        formatted_chunks = []
        for i, c in enumerate(chunks):
            source = c.get("source") or c.get("metadata", {}).get("source", "unknown")
            text = c.get("text", "")
            formatted_chunks.append(f"[{i+1}] Source: {source}\n{text}")
        context_block = "\n\n".join(formatted_chunks)

    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above. If there is not enough context provided just say so."
    )


def generate_answer(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Generates a final synthesized response using retrieved context chunks and Ollama.
    """
    if not chunks:
        return "I don't have enough information in the documents to answer that."

    prompt = build_prompt(query, chunks)

    try:
        response = ollama.chat(
            model=GEN_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.message.content.strip()  # type: ignore

    except Exception as e:
        print(f"[Warning] Ollama generation with '{GEN_MODEL}' failed ({e}). Returning context fallback.")
        sources = list(set(c.get("source") or c.get("metadata", {}).get("source", "unknown") for c in chunks))
        sources_str = ", ".join(sources)
        synthesis = f"Based on retrieved document context ({sources_str}):\n\n"
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source") or chunk.get("metadata", {}).get("source", "unknown")
            synthesis += f"--- Excerpt {i} ({source}) ---\n{chunk['text']}\n\n"
        return synthesis.strip()


if __name__ == "__main__":
    print("--- Testing Generator Module ---")
    mock_context = [{
        "text": "Meal expenses while traveling are reimbursed up to $60/day.",
        "metadata": {"source": "expense_policy.txt"}
    }]
    ans = generate_answer("What is the meal limit?", mock_context)
    print("Generated Output:")
    print(ans)

