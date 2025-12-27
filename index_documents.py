#!/usr/bin/env python3
"""
index_documents.py

Input: PDF or DOCX
- Extract clean text
- Split into chunks (fixed-overlap / sentence / paragraph)
- Create embeddings via Google Gemini API
- Store chunks + embeddings into PostgreSQL (pgvector)

Env vars required:
- GEMINI_API_KEY
- POSTGRES_URL  (SQLAlchemy URL, e.g. postgresql+psycopg://user:pass@host:5432/db)
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Literal, Sequence

from dotenv import load_dotenv

# ---------- Text extraction ----------
def extract_text_from_pdf(path: Path) -> str:
    # PyMuPDF is usually the most reliable for "clean enough" extraction.
    import fitz  # pymupdf
    doc = fitz.open(str(path))
    pages = []
    for i in range(len(doc)):
        pages.append(doc.load_page(i).get_text("text"))
    return "\n".join(pages)


def extract_text_from_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text is not None]
    return "\n".join(parts)


def clean_text(text: str) -> str:
    # basic cleaning: normalize whitespace, remove repeated blank lines
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------- Chunking strategies ----------
SplitStrategy = Literal["fixed", "sentence", "paragraph"]

def split_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras

def split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter without external NLP.
    Works decently for Hebrew/English mixed text.
    """
    # Split on end punctuation followed by whitespace/newline
    sents = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def chunk_fixed_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    chunk_size/overlap in characters (simple & deterministic).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == n:
            break
    return chunks


def split_to_chunks(text: str, strategy: SplitStrategy, chunk_size: int, overlap: int) -> List[str]:
    if strategy == "paragraph":
        return split_paragraphs(text)

    if strategy == "sentence":
        # Optional: group sentences into ~chunk_size chars (to avoid tiny chunks)
        sents = split_sentences(text)
        grouped: List[str] = []
        buf: List[str] = []
        buf_len = 0
        target = max(200, chunk_size)  # keep sane minimum
        for s in sents:
            if buf_len + len(s) + 1 > target and buf:
                grouped.append(" ".join(buf).strip())
                buf = [s]
                buf_len = len(s)
            else:
                buf.append(s)
                buf_len += len(s) + 1
        if buf:
            grouped.append(" ".join(buf).strip())
        return grouped

    if strategy == "fixed":
        return chunk_fixed_with_overlap(text, chunk_size=chunk_size, overlap=overlap)

    raise ValueError(f"Unknown strategy: {strategy}")


# ---------- Gemini embeddings ----------
@dataclass
class EmbeddingResult:
    vector: List[float]

def gemini_embed(texts: Sequence[str]) -> List[EmbeddingResult]:
    """
    Returns embeddings for each text in 'texts'.
    Uses google-genai SDK (recommended).
    """
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment")

    client = genai.Client(api_key=api_key)

    results: List[EmbeddingResult] = []
    # Use a common embedding model; if your course specifies another, swap it here.
    # Example model name often used for embeddings: "text-embedding-004"
    model_name = "text-embedding-004"

    for t in texts:
        resp = client.models.embed_content(
            model=model_name,
            contents=t
        )
        # The SDK response structure may vary by version; handle common shapes:
        vec = None
        if hasattr(resp, "embedding") and resp.embedding is not None:
            vec = resp.embedding.values
        elif hasattr(resp, "embeddings") and resp.embeddings:
            vec = resp.embeddings[0].values
        else:
            raise RuntimeError("Unexpected embedding response format from Gemini SDK")

        results.append(EmbeddingResult(vector=list(vec)))
    return results


# ---------- PostgreSQL storage (pgvector) ----------
def init_db_engine(postgres_url: str):
    from sqlalchemy import create_engine
    return create_engine(postgres_url, future=True)

def ensure_schema(engine):
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_chunks (
              id SERIAL PRIMARY KEY,
              chunk_text TEXT NOT NULL,
              embedding VECTOR(768) NOT NULL,
              filename TEXT NOT NULL,
              strategy_split TEXT NOT NULL,
              at_created TIMESTAMPTZ DEFAULT NOW()
            );
        """))

def insert_chunks(engine, filename: str, strategy: str, chunks: Sequence[str], embeddings: Sequence[EmbeddingResult]):
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have same length")

    from sqlalchemy import text

    now = datetime.now(timezone.utc)
    rows = []
    for chunk, emb in zip(chunks, embeddings):
        rows.append({
            "chunk_text": chunk,
            "embedding": emb.vector,
            "filename": filename,
            "strategy_split": strategy,
            "at_created": now,
        })

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO document_chunks (chunk_text, embedding, filename, strategy_split, at_created)
                VALUES (:chunk_text, :embedding, :filename, :strategy_split, :at_created)
            """),
            rows
        )


# ---------- Main ----------
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Index a PDF/DOCX into PostgreSQL with Gemini embeddings.")
    parser.add_argument("input_file", type=str, help="Path to input .pdf or .docx")
    parser.add_argument("--strategy", type=str, default="fixed", choices=["fixed", "sentence", "paragraph"])
    parser.add_argument("--chunk-size", type=int, default=1200, help="For fixed/sentence grouping: target chars per chunk")
    parser.add_argument("--overlap", type=int, default=200, help="For fixed strategy: overlap in chars")
    parser.add_argument("--batch-size", type=int, default=32, help="Insert batching (also helps memory)")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    ext = input_path.suffix.lower()
    if ext not in [".pdf", ".docx"]:
        raise ValueError("Input must be a .pdf or .docx")

    raw = extract_text_from_pdf(input_path) if ext == ".pdf" else extract_text_from_docx(input_path)
    text = clean_text(raw)
    if not text:
        raise RuntimeError("No text extracted from file")

    chunks = split_to_chunks(text, strategy=args.strategy, chunk_size=args.chunk_size, overlap=args.overlap)
    if not chunks:
        raise RuntimeError("No chunks produced")

    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise RuntimeError("Missing POSTGRES_URL in environment")

    engine = init_db_engine(postgres_url)
    ensure_schema(engine)

    # Embed & insert in batches
    total = 0
    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i:i + args.batch_size]
        embeds = gemini_embed(batch)
        insert_chunks(engine, filename=input_path.name, strategy=args.strategy, chunks=batch, embeddings=embeds)
        total += len(batch)
        print(f"Inserted {total}/{len(chunks)} chunks...")

    print(f"✅ Done. Inserted {len(chunks)} chunks for file={input_path.name}, strategy={args.strategy}")

if __name__ == "__main__":
    main()
