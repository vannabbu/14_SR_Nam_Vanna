# Local RAG Application (Homework Assignment)

A modular, offline Retrieval-Augmented Generation (RAG) system built in Python using **Ollama** (`nomic-embed-text` embeddings and `llama3.2:3b` LLM) and **ChromaDB** vector store.

---

##  Architecture & Modules

The project is structured under the `app/` package for clean separation of concerns:

- `app/config.py`: Centralized configuration (data paths, model names, chunk size, top_k, system prompt).
- `app/ingestion.py`: Loads raw document files (`.txt`, `.md`, `.pdf`).
- `app/chunking.py`: Splits documents into overlapping text chunks with source metadata.
- `app/embeddings.py`: Generates vector embeddings via Ollama `nomic-embed-text`.
- `app/vector_store.py`: Initializes and manages the local ChromaDB database index.
- `app/retriever.py`: Takes user queries, embeds them, and queries ChromaDB for top $k$ matching chunks.
- `app/generator.py`: Constructs grounded prompts and queries `llama3.2:3b` for response synthesis.
- `app/pipeline.py`: Orchestrates indexing (`run_indexing_pipeline`) and query handling (`run_rag_pipeline`).
- `app/main.py`: Interactive command-line interface (CLI) chat loop with `--reindex` support.

---

##  How to Run & Test

### 1. Interactive Chat Application
To start the chat application loop:
```bash
poetry run python -m app.main
```
*Options:* Pass `--reindex` to force re-ingestion of the `data/` folder.

### 2. Standalone Vector Check Script
To test the vector database retriever independently:
```bash
poetry run python demo_vector_check.py
```

### 3. 5-Question Benchmark Suite
To execute the automated benchmark evaluation:
```bash
poetry run python test_rag_benchmark.py
```

---

##  5-Question Benchmark Test Results

The RAG application was evaluated against 5 queries (4 in-domain IT support questions and 1 out-of-domain question).

### Question 1 [INSIDE DOCS]: "How do I set up my mobile device for company email?"
- **Retrieved Chunks:**
  1. `001_Setting_Up_a_Mobile_Device_for_Company_Email.txt` (Distance: 0.1965)
  2. `001_Setting_Up_a_Mobile_Device_for_Company_Email.txt` (Distance: 0.3362)
  3. `001_Setting_Up_a_Mobile_Device_for_Company_Email.txt` (Distance: 0.3802)
- **Final Answer:**
  > To set up your mobile device for company email, follow the steps below:
  > 1. Go to the Settings app on your mobile device.
  > 2. Select "Mail" or "Email" (depending on your device's operating system).
  > 3. Tap "Add Account" or "Create a new account".
  > 4. Select "Exchange" or "Corporate" as the account type.
  > 5. Enter your company email address and password.
  > 6. If prompted, enter the company's email server address (e.g., mail.company.com).
  > 7. Select the desired synchronization options (e.g., sync email, contacts, calendar).

### Question 2 [INSIDE DOCS]: "How do I configure VPN access for remote work?"
- **Retrieved Chunks:**
  1. `003_Configuring_VPN_Access_for_Remote_Workers.txt` (Distance: 0.2967)
  2. `003_Configuring_VPN_Access_for_Remote_Workers.txt` (Distance: 0.4661)
  3. `003_Configuring_VPN_Access_for_Remote_Workers.txt` (Distance: 0.5436)
- **Final Answer:**
  > Based on the provided context, you need to download and install the VPN client software from the company portal and launch it. For detailed configuration steps beyond installation, consult full IT support.

### Question 3 [INSIDE DOCS]: "What steps should I take if Microsoft Office is having issues?"
- **Retrieved Chunks:**
  1. `004_Troubleshooting_Issues_with_Microsoft_Office.txt` (Distance: 0.2133)
  2. `004_Troubleshooting_Issues_with_Microsoft_Office.txt` (Distance: 0.2749)
  3. `004_Troubleshooting_Issues_with_Microsoft_Office.txt` (Distance: 0.3323)
- **Final Answer:**
  > According to the troubleshooting guide, the first step is to restart Microsoft Office. Additional tips include checking for antivirus conflicts and running Office Repair tool.

### Question 4 [INSIDE DOCS]: "How do I reset a forgotten PIN code?"
- **Retrieved Chunks:**
  1. `002_Resetting_a_Forgotten_PIN.txt` (Distance: 0.3026)
  2. `002_Resetting_a_Forgotten_PIN.txt` (Distance: 0.4186)
  3. `002_Resetting_a_Forgotten_PIN.txt` (Distance: 0.4467)
- **Final Answer:**
  > To reset a forgotten PIN code:
  > 1. Access the PIN Reset Tool.
  > 2. Enter a new PIN (at least 8 characters with mixed case, numbers, and special characters).
  > 3. Re-enter the PIN to confirm and click "Reset PIN".

### Question 5 [NOT IN DOCS]: "What is the reimbursement limit for business travel meal expenses?"
- **Retrieved Chunks:**
  1. `001_Setting_Up_a_Mobile_Device_for_Company_Email.txt` (Distance: 1.0312)
  2. `001_Setting_Up_a_Mobile_Device_for_Company_Email.txt` (Distance: 1.0547)
  3. `001_Setting_Up_a_Mobile_Device_for_Company_Email.txt` (Distance: 1.0819)
- **Final Answer:**
  > I don't have enough information in the documents to answer that.

---

##  Reflection

### Overview
This project implements an offline Retrieval-Augmented Generation (RAG) system to evaluate local embedding and retrieval performance.

### What Worked Well
### Strengths
- **Modular Architecture:** Decoupled components (`ingestion`, `chunking`, `embeddings`, `vector_store`, `retriever`, `generator`) simplify maintenance and testing.
- **Retrieval Precision:** Local Ollama embeddings (`nomic-embed-text`) with ChromaDB achieved distance scores under 0.35 for relevant context snippets.
- **Context Grounding:** The LLM (`llama3.2:3b`) consistently answered in-domain queries and correctly rejected out-of-domain queries when relevance distance exceeded threshold metrics.

### Challenges
- **Context Fragmentation:** Fixed chunk sizes (400 characters) occasionally split continuous multi-step procedures across chunks, leading to partial context retrieval.
  
### Future Improvements
- **Re-Ranking:** Integrate a Cross-Encoder model (`bge-reranker-large`) to re-score candidate chunks before LLM generation.
- **Parent-Child Chunking:** Index smaller child chunks for retrieval while passing broader parent context to the generator.
