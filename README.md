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

Building this Retrieval-Augmented Generation (RAG) system provided valuable hands-on insight into modern local search and generation architectures.

### What Worked Well
The modular architecture (`ingestion`, `chunking`, `embeddings`, `vector_store`, `retriever`, and `generator`) made component isolation and testing straightforward. Utilizing Ollama with `nomic-embed-text` paired with ChromaDB yielded high similarity accuracy (distance scores below 0.35 for relevant chunks), while `llama3.2:3b` executed strict context-grounded synthesis. The system reliably rejected out-of-domain queries—such as questions about business travel meal reimbursements—by returning high distance metrics (>1.0) and adhering to the fallback prompt instruction: *"I don't have enough information in the documents to answer that."*

### What Was Harder Than Expected
Balancing chunk size and context overlap proved surprisingly tricky. Small fixed-size chunks (400 characters) occasionally split contiguous multi-step instructions (such as step-by-step VPN configuration guides) across chunk boundaries. This led to instances where the retriever returned partial context, causing the generator to notice incomplete steps and issue partial fallback responses despite relevant information existing across adjacent blocks.

### Future Improvement with Advanced RAG
To resolve context fragmentation and improve retrieval precision, I plan to implement **Re-Ranking** using a Cross-Encoder model (such as `bge-reranker-large`). Initial vector retrieval will fetch a broader top-$k$ candidate list (e.g., $k=10$), which the cross-encoder will re-score based on full query-document cross-attention before handing the top 3 snippets to the LLM. Combining re-ranking with **Parent-Child Chunking** will preserve macro-level context while allowing granular vector indexing.
