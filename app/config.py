"""
Configuration settings for RAG Pipeline.
"""

import os

# Project root directory (one level up from app/)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

# Base paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

# Vector DB Settings
COLLECTION_NAME = "rag_homework"

# Models
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.2:3b"

# Chunking Settings
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Retrieval Settings
TOP_K = 3

# Generation Settings
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context provided below. If the answer is not contained in the context, "
    "say \"I don't have enough information in the documents to answer that.\" "
    "Do not use outside knowledge. Cite the source file name(s) you used."
)

