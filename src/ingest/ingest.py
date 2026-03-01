# src/ingest/ingest_pipeline.py
# ============================================================================
# Ingestion Pipeline — Section-Aware Chunking with LLM Header Extraction
# Parses PDFs, identifies logical sections via Claude, chunks by section,
# builds FAISS vector index + BM25 sparse index for hybrid retrieval.
# ============================================================================

import os
import sys
import json
import time
import pickle
import logging
import argparse
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()  # loads ANTHROPIC_API_KEY from .env

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not found.")
    print("Create a .env file in the repo root with: ANTHROPIC_API_KEY=sk-ant-...")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/ingest.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# Model and chunking parameters
LLM_MODEL = "claude-haiku-4-5-20251001"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CONTEXT_CHARS = 60_000  # truncation limit for header extraction prompt
API_CALL_DELAY = 2.0  # seconds between LLM calls to avoid rate limits (1s was insufficient)

# Paths (relative to repo root)
DATA_MANIFEST = "data/data_manifest.csv"
RAW_DIR = "data/raw"
INDEX_DIR = "data/processed/faiss_index"
BM25_PATH = "data/processed/bm25_corpus.pkl"


# ---------------------------------------------------------------------------
# Pydantic schema for structured LLM output
# ---------------------------------------------------------------------------
class PaperOutline(BaseModel):
    headers: list[str] = Field(
        description="Chronological list of exact section/subsection titles from the paper."
    )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda rs: logger.warning(
        f"LLM call failed (attempt {rs.attempt_number}), retrying..."
    ),
)
def extract_sections_with_llm(raw_text: str, llm: ChatAnthropic) -> list[str]:
    """Uses Claude to extract logical section headers from a paper."""
    structured_llm = llm.with_structured_output(PaperOutline)
    prompt = (
        "You are an expert academic parser. Review the following text from an "
        "academic paper. Extract a chronological list of all major Section and "
        "Subsection titles exactly as they appear in the text "
        '(e.g., "1. Introduction", "II. Proposed Methodology", "3.1 Data Collection").\n\n'
        f"Paper Text (Truncated):\n{raw_text[:MAX_CONTEXT_CHARS]}"
    )
    result = structured_llm.invoke(prompt)
    return result.headers


def split_text_by_headers(text: str, headers: list[str]) -> list[dict]:
    """Splits raw text into sections based on exact header string matches."""
    sections = []
    last_idx = 0
    current_header = "Front Matter / Abstract"

    for header in headers:
        idx = text.find(header, last_idx)
        if idx != -1:
            section_content = text[last_idx:idx].strip()
            if section_content:
                sections.append({"header": current_header, "content": section_content})
            last_idx = idx + len(header)
            current_header = header

    final_content = text[last_idx:].strip()
    if final_content:
        sections.append({"header": current_header, "content": final_content})

    return sections


def build_bm25_corpus(all_chunks) -> None:
    """Persists chunk texts + metadata for BM25 sparse retrieval at query time."""
    corpus = []
    for chunk in all_chunks:
        corpus.append({
            "chunk_id": chunk.metadata.get("chunk_id", "UNKNOWN"),
            "text": chunk.page_content,
            "metadata": dict(chunk.metadata),
        })
    os.makedirs(os.path.dirname(BM25_PATH), exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(corpus, f)
    logger.info(f"BM25 corpus saved ({len(corpus)} chunks) → {BM25_PATH}")


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------
def run_llm_section_aware_ingestion(force: bool = False):
    """End-to-end ingestion: PDF → section-aware chunks → FAISS + BM25 indices."""

    os.makedirs("logs", exist_ok=True)

    # --- Skip if indices already exist (unless --force) ---
    faiss_exists = os.path.isdir(INDEX_DIR) and os.listdir(INDEX_DIR)
    bm25_exists = os.path.exists(BM25_PATH)

    if faiss_exists and bm25_exists and not force:
        logger.info(
            f"FAISS index ({INDEX_DIR}) and BM25 corpus ({BM25_PATH}) already exist. "
            "Skipping ingestion. Use --force to rebuild."
        )
        return

    if force:
        logger.info("--force flag set. Rebuilding indices from scratch.")

    if not os.path.exists(DATA_MANIFEST):
        logger.error(f"Data manifest not found at {DATA_MANIFEST}")
        sys.exit(1)

    df = pd.read_csv(DATA_MANIFEST)
    df.fillna("", inplace=True)
    logger.info(f"Loaded manifest with {len(df)} sources.")

    # Configure tools
    try:
        llm = ChatAnthropic(model=LLM_MODEL, temperature=0, max_tokens=1024)
    except Exception as e:
        logger.error(f"Failed to initialize Claude LLM: {e}")
        sys.exit(1)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    all_chunks = []
    processing_log = []

    for _, row in df.iterrows():
        source_id = row["source_id"]
        pdf_path = os.path.join(RAW_DIR, row["raw_path"])

        if not os.path.exists(pdf_path):
            logger.warning(f"Skipping {source_id}: {pdf_path} not found.")
            continue

        logger.info(f"Processing {source_id}...")

        # --- PDF Loading (with error handling) ---
        try:
            loader = PyMuPDFLoader(pdf_path)
            pages = loader.load()
            raw_full_text = "\n".join([p.page_content for p in pages])
        except Exception as e:
            logger.error(f"  PDF load failed for {source_id}: {e}")
            continue

        # --- LLM Header Extraction (with retry via decorator) ---
        try:
            extracted_headers = extract_sections_with_llm(raw_full_text, llm)
            logger.info(f"  Found {len(extracted_headers)} headers.")
            time.sleep(API_CALL_DELAY)  # proactive rate-limit throttle
        except Exception as e:
            logger.error(f"  Header extraction failed for {source_id} after retries: {e}")
            extracted_headers = []

        logical_sections = split_text_by_headers(raw_full_text, extracted_headers)

        # --- Chunking ---
        document_chunks = []
        for section in logical_sections:
            sub_chunks = text_splitter.create_documents([section["content"]])
            for chunk in sub_chunks:
                chunk.metadata.update({
                    "source_id": row["source_id"],
                    "title": row["title"],
                    "year": str(row["year"]),
                    "tags": row.get("tags", ""),
                    "source_type": row.get("source_type", ""),
                    "Section": section["header"],
                })
                document_chunks.append(chunk)

        for i, chunk in enumerate(document_chunks):
            chunk.metadata["chunk_id"] = f"{source_id}_chunk_{i}"

        all_chunks.extend(document_chunks)
        logger.info(f"  Generated {len(document_chunks)} chunks.")
        processing_log.append({
            "source_id": source_id,
            "headers_found": len(extracted_headers),
            "chunks_created": len(document_chunks),
        })

    logger.info(f"\nTotal section-aware chunks across corpus: {len(all_chunks)}")

    if not all_chunks:
        logger.error("No chunks created. Check your data/raw/ directory and manifest.")
        sys.exit(1)

    # --- Build FAISS vector index ---
    logger.info("Building FAISS index (this may take a minute)...")
    try:
        os.makedirs(os.path.dirname(INDEX_DIR), exist_ok=True)
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        vectorstore.save_local(INDEX_DIR)
        logger.info(f"FAISS index saved → {INDEX_DIR}")
    except Exception as e:
        logger.error(f"FAISS indexing failed: {e}")
        sys.exit(1)

    # --- Build BM25 sparse corpus for hybrid retrieval ---
    build_bm25_corpus(all_chunks)

    # --- Save processing log ---
    log_path = "logs/ingest_log.json"
    with open(log_path, "w") as f:
        json.dump(processing_log, f, indent=2)
    logger.info(f"Ingestion log saved → {log_path}")

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDFs into FAISS + BM25 indices.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild indices from scratch even if they already exist.",
    )
    args = parser.parse_args()
    run_llm_section_aware_ingestion(force=args.force)
