# Document Vector Indexer (Gemini + PostgreSQL)

## Author
Ariel Gurten

## Overview

This project implements a **document indexing and vectorization pipeline** using
**Google Gemini embeddings** and **PostgreSQL with pgvector**.

The system:
- Extracts clean text from PDF and DOCX documents
- Splits text into chunks using configurable strategies
- Generates embeddings for each chunk using Google Gemini API
- Stores chunks, vectors, and metadata in PostgreSQL
- Designed to support similarity search and RAG-style pipelines

---

## Project Structure

```
Document-Vector-Indexer/
├── index_documents.py      
├── requirements.txt
├── README.md
├── .env                    # Environment variables (NOT committed)
```

---

## Requirements

- Python 3.10+
- PostgreSQL 15+
- pgvector extension
- Google Gemini API key

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/ArielGurten/Document-Vector-Indexer/
cd Document-Vector-Indexer
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## PostgreSQL Setup

### 1. Create a database
```sql
CREATE DATABASE db_ex;
```

### 2. Enable pgvector extension
```sql
CREATE EXTENSION vector;
```

> The script will automatically create the required table if it does not exist.

---

## Database Schema

The script creates and uses the following table structure (as required):

```sql
CREATE TABLE document_chunks (
  id SERIAL PRIMARY KEY,
  chunk_text TEXT NOT NULL,
  embedding VECTOR(768) NOT NULL,
  filename TEXT NOT NULL,
  split_strategy TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY= your_gemini_api_key
POSTGRES_URL=postgresql+psycopg://user:password@localhost:5432/db_ex
```

---

## Indexing Documents

### Script
```
index_documents.py
```

### Supported Input Formats
- PDF
- DOCX

### Chunking Strategies
- `fixed` - fixed-size chunks with overlap
- `sentence` - sentence-based splitting
- `paragraph` - paragraph-based splitting

### Command-Line Options

```bash
python index_documents.py <input_file> \
  --strategy fixed|sentence|paragraph \
  --chunk-size 1200 \
  --overlap 200 \
  --batch-size 32 \
  --embedding-model gemini-embedding-001 \
  --embedding-dim 768
```

### Example Usage

```bash
python index_documents.py sample.pdf --strategy fixed --chunk-size 1200 --overlap 200
```

```bash
python index_documents.py sample.docx --strategy sentence
```

---

## Embeddings

- Embedding model: **gemini-embedding-001**
- Embedding dimensionality: **768**
- Embeddings are generated in batches for efficiency
- Each chunk is embedded independently and stored with metadata

---

## Technologies Used

- Python
- Google Gemini API
- PostgreSQL
- pgvector
- SQLAlchemy
- PyMuPDF
- python-docx
- python-dotenv

---

## Notes

- The system is modular and extensible
- Designed for use in semantic search and RAG pipelines
- Clean, deterministic chunking and embedding workflow
