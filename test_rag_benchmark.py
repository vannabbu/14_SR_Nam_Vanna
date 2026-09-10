"""
Automated 5-Question Test Suite for RAG Homework.

Tests:
 - 4 Questions with answers INSIDE the documents
 - 1 Question with answer NOT inside the documents
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.pipeline import run_rag_pipeline


def main():
    test_questions = [
        {
            "id": 1,
            "type": "INSIDE DOCS",
            "question": "How do I set up my mobile device for company email?"
        },
        {
            "id": 2,
            "type": "INSIDE DOCS",
            "question": "How do I configure VPN access for remote work?"
        },
        {
            "id": 3,
            "type": "INSIDE DOCS",
            "question": "What steps should I take if Microsoft Office is having issues?"
        },
        {
            "id": 4,
            "type": "INSIDE DOCS",
            "question": "How do I reset a forgotten PIN code?"
        },
        {
            "id": 5,
            "type": "NOT IN DOCS",
            "question": "What is the reimbursement limit for business travel meal expenses?"
        }
    ]

    print("=======================================================================")
    print("           RAG Application 5-Question Benchmark Test                  ")
    print("=======================================================================\n")

    report_lines = []

    for item in test_questions:
        q_num = item["id"]
        q_type = item["type"]
        query = item["question"]

        print(f"Executing Test Question #{q_num} [{q_type}]")
        print(f"Question: \"{query}\"")

        answer, retrieved_chunks = run_rag_pipeline(query, top_k=3)

        print("\nRetrieved Context Chunks:")
        chunk_info_list = []
        for idx, c in enumerate(retrieved_chunks, 1):
            source = c.get("source") or c.get("metadata", {}).get("source", "unknown")
            dist = c.get("distance", 0.0)
            text_preview = c.get("text", "")[:120].replace("\n", " ")
            chunk_str = f"  [{idx}] Source: {source} (Distance: {dist:.4f})\n      Snippet: \"{text_preview}...\""
            print(chunk_str)
            chunk_info_list.append(chunk_str)

        print(f"\nFinal Generated Answer:\n{answer}\n")
        print("=" * 70 + "\n")

        # Format for summary report
        report_lines.append(f"### Question {q_num} [{q_type}]: \"{query}\"")
        report_lines.append("**Retrieved Chunks:**")
        for chunk_info in chunk_info_list:
            report_lines.append(f"- {chunk_info.strip()}")
        report_lines.append(f"\n**Final Answer:**\n> {answer}\n")
        report_lines.append("-" * 50 + "\n")

    # Save results to a file for reference
    report_file = os.path.join(PROJECT_ROOT, "test_results.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Test suite complete! Results saved to '{report_file}'.")


if __name__ == "__main__":
    main()
