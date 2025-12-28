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
- POSTGRES_URL  (SQLAlchemy URL, e.g. postgresql+psycopg://user:password@host:5432/db)

DB columns (per assignment):
- id (unique)
- chunk_text
- embedding
- filename
- split_strategy
- created_at
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Sequence

from dotenv import load_dotenv

# ---------- Text extraction ----------
def extract_text_from_pdf(path: Path) -> str:
    import fitz  # pymupdf
    doc = fitz.open(str(path))
    try:
        pages = []
        for i in range(len(doc)):
            pages.append(doc.load_page(i).get_text("text"))
        return "\n".join(pages)
    finally:
        doc.close()



def extract_text_from_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text]
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
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter without external NLP.
    """
    sents = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]

def chunk_fixed_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    chunk_size/overlap in characters.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def split_to_chunks(text: str, strategy: SplitStrategy, chunk_size: int, overlap: int) -> List[str]:
    if strategy == "paragraph":
        return split_paragraphs(text)

    if strategy == "sentence":
        # Group sentences into ~chunk_size chars to avoid tiny chunks
        sents = split_sentences(text)
        grouped: List[str] = []
        buf: List[str] = []
        buf_len = 0
        target = max(200, chunk_size)

        for s in sents:
            if buf and (buf_len + len(s) + 1 > target):
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

def gemini_embed(texts: Sequence[str], *, model: str, output_dimensionality: int, max_retries: int = 5) -> List[EmbeddingResult]:
    """
    Returns embeddings for each text in 'texts' (batched in a single request).
    Uses the Google GenAI SDK.
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment")

    client = genai.Client(api_key=api_key)

    # Retry on transient failures (rate limit / 5xx)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.embed_content(
                model=model,
                contents=list(texts),
                config=types.EmbedContentConfig(output_dimensionality=output_dimensionality),
            )
            if not hasattr(resp, "embeddings") or not resp.embeddings:
                raise RuntimeError("Unexpected embedding response format (missing embeddings)")

            results: List[EmbeddingResult] = []
            for emb_obj in resp.embeddings:
                vec = getattr(emb_obj, "values", None)
                if vec is None:
                    raise RuntimeError("Unexpected embedding object format (missing values)")
                results.append(EmbeddingResult(vector=list(vec)))

            if len(results) != len(texts):
                raise RuntimeError(
                    f"Embedding count mismatch: got {len(results)} results for {len(texts)} inputs"
                )

            # Safety check: ensure Gemini returned vectors with the expected dimension
            bad_dims = [len(r.vector) for r in results if len(r.vector) != output_dimensionality]
            if bad_dims:
                raise RuntimeError(
                    f"Embedding dimensionality mismatch. Expected {output_dimensionality}, got {bad_dims[:5]}..."
                )

            return results

        except Exception as e:
            last_err = e
            sleep_s = min(2 ** attempt, 16)
            time.sleep(sleep_s)

    raise RuntimeError(f"Gemini embedding failed after {max_retries} retries: {last_err}")


# ---------- PostgreSQL storage (pgvector) ----------
def init_db_engine(postgres_url: str):
    from sqlalchemy import create_engine
    return create_engine(postgres_url, future=True)

def ensure_schema(engine, *, vector_dim: int):
    """
    Creates pgvector extension and the document_chunks table with assignment-required column names.
    """
    from sqlalchemy import MetaData, Table, Column, Integer, Text, DateTime
    from sqlalchemy import text as sql_text
    from pgvector.sqlalchemy import Vector

    with engine.begin() as conn:
        conn.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector;"))

    metadata = MetaData()
    Table(
        "document_chunks",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("chunk_text", Text, nullable=False),
        Column("embedding", Vector(vector_dim), nullable=False),
        Column("filename", Text, nullable=False),
        Column("split_strategy", Text, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
    )
    metadata.create_all(engine)

def insert_chunks(engine, *, filename: str, split_strategy: str, chunks: Sequence[str], embeddings: Sequence[EmbeddingResult]):
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have same length")

    from sqlalchemy import MetaData, Table, insert

    metadata = MetaData()
    table = Table("document_chunks", metadata, autoload_with=engine)

    now = datetime.now(timezone.utc)
    rows = []
    for chunk, emb in zip(chunks, embeddings):
        rows.append(
            {
                "chunk_text": chunk,
                "embedding": emb.vector,         
                "filename": filename,
                "split_strategy": split_strategy,
                "created_at": now,
            }
        )

    with engine.begin() as conn:
        conn.execute(insert(table), rows)


# ---------- Main ----------
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Index a PDF/DOCX into PostgreSQL with Gemini embeddings.")
    parser.add_argument("input_file", type=str, help="Path to input .pdf or .docx")
    parser.add_argument("--strategy", type=str, default="fixed", choices=["fixed", "sentence", "paragraph"])
    parser.add_argument("--chunk-size", type=int, default=1200, help="For fixed/sentence grouping: target chars per chunk")
    parser.add_argument("--overlap", type=int, default=200, help="For fixed strategy: overlap in chars")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding + insert batching")
    parser.add_argument("--embedding-model", type=str, default="gemini-embedding-001", help="Gemini embedding model name")
    parser.add_argument("--embedding-dim", type=int, default=768, help="Embedding dimensionality (must match DB VECTOR(dim))")
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
    ensure_schema(engine, vector_dim=args.embedding_dim)

    total = 0
    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i:i + args.batch_size]
        embeds = gemini_embed(
            batch,
            model=args.embedding_model,
            output_dimensionality=args.embedding_dim,
        )
        insert_chunks(
            engine,
            filename=input_path.name,
            split_strategy=args.strategy,
            chunks=batch,
            embeddings=embeds,
        )
        total += len(batch)
        print(f"Inserted {total}/{len(chunks)} chunks...")

    print(f"Done. Inserted {len(chunks)} chunks for file={input_path.name}, split_strategy={args.strategy}")

if __name__ == "__main__":
    main()
