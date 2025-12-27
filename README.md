# Document Vector Indexer (Gemini + PostgreSQL)

## Author

Ariel Gurten

This project implements a **document vectorization and retrieval pipeline** using **Google Gemini embeddings** and **PostgreSQL with pgvector**.

The system:
- Extracts text from PDF/DOCX documents
- Splits text into chunks using multiple strategies
- Generates embeddings for each chunk using Gemini
- Stores vectors and metadata in PostgreSQL
- Supports similarity search and RAG-style question answering

---

## Project Structure

```
Document-Vector-Indexer/
├── index_documents.py      # Part 2 – document ingestion & vector creation
├── test.py                 # Part 4 – vector similarity search
├── rag_query.py            # RAG-style retrieval + generation
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
git clone <YOUR_GITHUB_REPO_URL>
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

### 2. Enable pgvector
```sql
CREATE EXTENSION vector;
```

### 3. Create the table
```sql
CREATE TABLE document_chunks (
  id SERIAL PRIMARY KEY,
  chunk_text TEXT NOT NULL,
  embedding VECTOR(768) NOT NULL,
  filename TEXT NOT NULL,
  strategy_split TEXT NOT NULL,
  at_created TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
POSTGRES_URL=postgresql+psycopg://postgres:password@localhost:5432/db_ex
```

---

## Part 2 – Indexing Documents

### Script
```
index_documents.py
```

### Supported input
- PDF
- DOCX

### Chunking strategies
- `fixed` – fixed-size chunks with overlap
- `sentence` – sentence-based splitting
- `paragraph` – paragraph-based splitting

### Example usage

```bash
python index_documents.py sample.pdf --strategy fixed --chunk-size 1200 --overlap 200
```

```bash
python index_documents.py sample.docx --strategy sentence
```

Each chunk is embedded using **Google Gemini** (`text-embedding-004`) and stored in PostgreSQL.

---

## Part 4 – Vector Similarity Search

### Script
```
test.py
```

### Run
```bash
python test.py
```

You will be prompted to enter a query.
The script:
1. Embeds the query using Gemini
2. Searches the database using pgvector similarity (`<->`)
3. Returns the most similar chunks

---

## RAG – Retrieval Augmented Generation (Bonus)

### Script
```
rag_query.py
```

### Run
```bash
python rag_query.py
```

### What it does
1. Embeds the user question
2. Retrieves the top-K relevant chunks from PostgreSQL
3. Sends the chunks as **context** to Gemini
4. Generates an answer **based only on retrieved content**

This implements a full **RAG (Retrieval-Augmented Generation)** pipeline.

---

## Technologies Used

- Python
- PostgreSQL
- pgvector
- Google Gemini API
- SQLAlchemy
- PyMuPDF
- python-docx
- python-dotenv

---

## Security Considerations

- No secrets stored in source code
- API keys managed via `.env`
- Database credentials not hardcoded

---

## Notes

- Embedding model: `text-embedding-004`
- Generation model (RAG): `gemini-1.5-pro`
- Vector dimension: 768

---
