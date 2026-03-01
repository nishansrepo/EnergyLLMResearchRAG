# src/rag/rag.py
# ============================================================================
# RAG Pipeline — Baseline vs. Enhanced (3 Enhancements) with Session Caching
#
# Enhancements over baseline (top-k vector search):
#   1. Cross-Encoder Reranking  (ms-marco-MiniLM-L-6-v2)
#   2. BM25 Hybrid Retrieval    (Reciprocal Rank Fusion of sparse + dense)
#   3. Metadata Filtering        (year / source_type faceted retrieval)
#
# Session caching: every run is persisted to logs/session_cache.json so
# graders can replay the demo without an API key.
# ============================================================================

import os
import sys
import json
import time
import pickle
import logging
import argparse
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/rag.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# Model config
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_MAX_TOKENS = 1024
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval parameters
BASELINE_K = 4
ENHANCED_VECTOR_K = 15
ENHANCED_BM25_K = 15
FINAL_TOP_K = 4
RRF_K_CONSTANT = 60  # Reciprocal Rank Fusion constant
DECOMPOSE_VECTOR_K = 8  # per-sub-query retrieval depth for decomposed queries
DECOMPOSE_BM25_K = 8
API_CALL_DELAY = 2.0  # seconds between LLM calls to avoid rate limits

# Paths
FAISS_INDEX_DIR = "data/processed/faiss_index"
BM25_CORPUS_PATH = "data/processed/bm25_corpus.pkl"
SESSION_CACHE_PATH = "logs/session_cache.json"
RUN_LOG_PATH = "logs/run_logs.json"

# Prompt
PROMPT_VERSION = "v3.1-hybrid-decompose-citations"
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert AI research assistant for the energy sector.
Your task is to answer the user's query using ONLY the provided context chunks.

TRUST BEHAVIOR & CONSTRAINTS:
1. If the context does not contain the answer, explicitly state exactly:
   "I cannot find evidence for this claim in the provided context."
   Do NOT hallucinate or invent information.
2. If the context contains CONFLICTING evidence across sources, flag it:
   "Conflicting evidence found:" and cite both sides.
3. Every claim or fact you output MUST be cited.
4. Your citation MUST contain both the exact Chunk ID and a short, direct text
   snippet from the context that supports the claim.
5. Format your final answer as a Markdown table with the following columns:
   | Claim / Information | Supporting Text Snippet | Chunk ID |

Context:
{context}"""),
    ("human", "{question}"),
])


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _ensure_dirs():
    """Create required directories if they don't exist."""
    for d in ["logs", "outputs", "data/processed"]:
        os.makedirs(d, exist_ok=True)


def format_docs(docs) -> str:
    """Formats retrieved LangChain Documents for the LLM context window."""
    parts = []
    for doc in docs:
        cid = doc.metadata.get("chunk_id", "UNKNOWN_ID")
        src = doc.metadata.get("source_id", "")
        yr = doc.metadata.get("year", "")
        sec = doc.metadata.get("Section", "")
        header = f"--- Chunk ID: [{cid}] | Source: {src} | Year: {yr} | Section: {sec} ---"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def _doc_to_dict(doc) -> dict:
    """Serialize a LangChain Document to a plain dict (for JSON caching)."""
    return {
        "page_content": doc.page_content,
        "metadata": dict(doc.metadata),
    }


# ---------------------------------------------------------------------------
# BM25 Hybrid Retrieval (Enhancement #2)
# ---------------------------------------------------------------------------
class BM25Index:
    """Wraps rank_bm25 over the pre-built corpus for sparse retrieval."""

    def __init__(self, corpus_path: str = BM25_CORPUS_PATH):
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"BM25 corpus not found at {corpus_path}. Run ingestion first."
            )
        with open(corpus_path, "rb") as f:
            self._corpus = pickle.load(f)

        tokenized = [entry["text"].lower().split() for entry in self._corpus]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index loaded ({len(self._corpus)} documents).")

    def search(self, query: str, k: int = ENHANCED_BM25_K) -> list[dict]:
        """Return top-k results as list of {chunk_id, text, metadata, score}."""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for idx, score in ranked:
            entry = self._corpus[idx]
            results.append({
                "chunk_id": entry["chunk_id"],
                "text": entry["text"],
                "metadata": entry["metadata"],
                "bm25_score": float(score),
            })
        return results


def reciprocal_rank_fusion(
    vector_docs: list, bm25_results: list[dict], k: int = RRF_K_CONSTANT
) -> list:
    """
    Fuse FAISS vector results and BM25 sparse results using RRF.
    Returns LangChain Document objects sorted by fused score.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, object] = {}

    # Score vector results
    for rank, doc in enumerate(vector_docs):
        cid = doc.metadata.get("chunk_id", "UNKNOWN")
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        doc_map[cid] = doc

    # Score BM25 results
    for rank, entry in enumerate(bm25_results):
        cid = entry["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        if cid not in doc_map:
            # Create a lightweight LangChain-compatible object
            from langchain_core.documents import Document
            doc_map[cid] = Document(
                page_content=entry["text"], metadata=entry["metadata"]
            )

    # Sort by fused score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[cid] for cid, _ in ranked]


# ---------------------------------------------------------------------------
# Metadata Filtering (Enhancement #3)
# ---------------------------------------------------------------------------
def apply_metadata_filters(
    docs: list,
    year_filter: Optional[str] = None,
    source_type_filter: Optional[str] = None,
) -> list:
    """
    Post-retrieval metadata filtering on year and/or source_type.
    Filters are optional; None means no filtering on that dimension.
    """
    filtered = docs
    if year_filter:
        filtered = [
            d for d in filtered
            if str(d.metadata.get("year", "")) == str(year_filter)
        ]
    if source_type_filter:
        filtered = [
            d for d in filtered
            if source_type_filter.lower() in str(d.metadata.get("source_type", "")).lower()
        ]
    if not filtered:
        logger.warning("Metadata filters removed ALL results; falling back to unfiltered.")
        return docs
    return filtered


# ---------------------------------------------------------------------------
# Query Decomposition for Multi-hop Synthesis (Enhancement #4)
# ---------------------------------------------------------------------------
# Synthesis queries like "Compare A with B" fail when retrieval gravitates
# to one side. Decomposition splits the query into sub-queries, retrieves
# for each independently, and merges the candidate pools before reranking.

SYNTHESIS_SIGNALS = [
    "compare", "contrast", "synthesize", "both", "align", "differ",
    "agree", "disagree", "versus", "vs.", "vs ", "how do", "how does",
    "complicate", "interact", "relate", "consensus",
]

# Patterns that indicate multiple sources are referenced in the query
MULTI_SOURCE_PATTERNS = [
    # Two "et al." references
    r"(?i)\b\w+\s+et\s+al\.?\s*.{1,120}?\b\w+\s+et\s+al\.?",
    # Two paper/study/survey/review mentions (wider gap)
    r"(?i)(?:paper|study|survey|review)\s+.{1,120}?(?:paper|study|survey|review)",
    # "et al." + named system/architecture (e.g., "Li et al. ... GAIA architecture")
    r"(?i)\b\w+\s+et\s+al\.?\s*.{1,120}?(?:architecture|model|framework|system|approach)",
    # Named system + "et al." (reverse order)
    r"(?i)(?:architecture|model|framework|system|paper)\s+.{1,120}?\b\w+\s+et\s+al\.?",
]


def _is_synthesis_query(query: str) -> bool:
    """Lightweight heuristic: does this query reference multiple sources or
    require cross-document reasoning?"""
    import re
    q_lower = query.lower()

    has_signal = any(s in q_lower for s in SYNTHESIS_SIGNALS)
    has_multi_source = any(re.search(p, query) for p in MULTI_SOURCE_PATTERNS)

    return has_signal and has_multi_source


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=2, min=4, max=20),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda rs: logger.warning(
        f"Decomposition LLM call failed (attempt {rs.attempt_number}), retrying..."
    ),
)
def _llm_decompose(query: str, llm) -> list[str]:
    """Use the LLM to split a synthesis query into focused sub-queries."""
    decompose_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research query planner. Given a complex comparison or synthesis "
         "question, decompose it into 2-3 focused sub-queries that each target a "
         "single source or concept. Output ONLY a JSON array of strings, nothing else.\n"
         "Example input: 'Compare the deployment challenges in EnergyGPT with "
         "the computational cost limitations in Sarwar et al.'\n"
         'Example output: ["What are the deployment infrastructure challenges '
         'discussed in the EnergyGPT paper?", "What computational cost limitations '
         'does the survey by Sarwar et al. highlight?"]'),
        ("human", "{query}"),
    ])
    chain = decompose_prompt | llm | StrOutputParser()
    raw = chain.invoke({"query": query})

    # Parse the JSON array from the response
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    sub_queries = json.loads(raw)

    if not isinstance(sub_queries, list) or len(sub_queries) < 2:
        raise ValueError(f"Expected list of 2+ sub-queries, got: {sub_queries}")

    return sub_queries


def decompose_and_retrieve(
    query: str,
    vectorstore,
    bm25_index: Optional[BM25Index],
    llm,
) -> list:
    """
    Decompose a synthesis query, retrieve per sub-query, and merge via RRF.
    Falls back to standard single-query retrieval on failure.
    """
    try:
        sub_queries = _llm_decompose(query, llm)
        logger.info(f"  Decomposed into {len(sub_queries)} sub-queries:")
        for i, sq in enumerate(sub_queries):
            logger.info(f"    [{i+1}] {sq}")
    except Exception as e:
        logger.warning(f"  Decomposition failed ({e}); using original query.")
        return None  # signal caller to fall back

    # Retrieve per sub-query
    all_vector_docs = []
    all_bm25_results = []

    for sq in sub_queries:
        v_docs = vectorstore.similarity_search(sq, k=DECOMPOSE_VECTOR_K)
        all_vector_docs.extend(v_docs)

        if bm25_index:
            b_results = bm25_index.search(sq, k=DECOMPOSE_BM25_K)
            all_bm25_results.extend(b_results)

    # Deduplicate by chunk_id before fusion (keep first occurrence)
    seen_ids = set()
    deduped_vector = []
    for doc in all_vector_docs:
        cid = doc.metadata.get("chunk_id", "")
        if cid not in seen_ids:
            seen_ids.add(cid)
            deduped_vector.append(doc)

    seen_ids_bm25 = set()
    deduped_bm25 = []
    for entry in all_bm25_results:
        cid = entry["chunk_id"]
        if cid not in seen_ids_bm25:
            seen_ids_bm25.add(cid)
            deduped_bm25.append(entry)

    logger.info(
        f"  Multi-query retrieval: {len(deduped_vector)} vector + "
        f"{len(deduped_bm25)} BM25 unique candidates."
    )

    # Fuse with RRF
    if bm25_index and deduped_bm25:
        fused = reciprocal_rank_fusion(deduped_vector, deduped_bm25)
    else:
        fused = deduped_vector

    return fused


# ---------------------------------------------------------------------------
# Session Caching / Replay
# ---------------------------------------------------------------------------
def _load_session_cache() -> list[dict]:
    if os.path.exists(SESSION_CACHE_PATH):
        with open(SESSION_CACHE_PATH, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def _save_session_cache(cache: list[dict]):
    with open(SESSION_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _find_cached_entry(query: str, cache: list[dict]) -> Optional[dict]:
    """Find the most recent cached entry for this exact query."""
    for entry in reversed(cache):
        if entry.get("query") == query:
            return entry
    return None


def _cache_session(entry: dict):
    """Append a session entry to the cache file."""
    cache = _load_session_cache()
    cache.append(entry)
    _save_session_cache(cache)


# ---------------------------------------------------------------------------
# LLM invocation with retry
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda rs: logger.warning(
        f"LLM generation failed (attempt {rs.attempt_number}), retrying..."
    ),
)
def _invoke_chain(chain, context: str, question: str) -> str:
    return chain.invoke({"context": context, "question": question})


# ---------------------------------------------------------------------------
# Run-log persistence (Phase 2 compatible)
# ---------------------------------------------------------------------------
def _append_run_log(entry: dict):
    logs = []
    if os.path.exists(RUN_LOG_PATH):
        with open(RUN_LOG_PATH, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    logs.append(entry)
    with open(RUN_LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)


# ---------------------------------------------------------------------------
# Main comparative pipeline
# ---------------------------------------------------------------------------
def run_comparative_rag(
    query: str,
    year_filter: Optional[str] = None,
    source_type_filter: Optional[str] = None,
    replay: bool = False,
) -> dict:
    """
    Runs both Baseline and Enhanced RAG pipelines, logs results, caches session.

    Parameters
    ----------
    query : str
        The user's research question.
    year_filter : str, optional
        Filter retrieved chunks to a specific publication year.
    source_type_filter : str, optional
        Filter retrieved chunks to a specific source type (e.g. "journal").
    replay : bool
        If True, return cached results instead of calling the LLM.

    Returns
    -------
    dict with keys: timestamp, query, baseline_*, enhanced_*, prompt_version
    """
    _ensure_dirs()

    # ----- Replay mode: return cached result if available -----
    if replay:
        cache = _load_session_cache()
        cached = _find_cached_entry(query, cache)
        if cached:
            logger.info(f"[REPLAY] Returning cached result for: {query[:60]}...")
            return cached
        logger.warning("[REPLAY] No cached entry found; running live pipeline.")

    # ----- Check API key for live runs -----
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error(
            "ANTHROPIC_API_KEY not set. Use a .env file or set it in your shell.\n"
            "For replay of cached sessions, pass --replay."
        )
        sys.exit(1)

    logger.info(f"Processing query: {query[:80]}...")

    # 1. Load indices
    logger.info("Loading FAISS index...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        sys.exit(1)

    logger.info("Loading BM25 index...")
    try:
        bm25_index = BM25Index()
    except FileNotFoundError as e:
        logger.warning(f"BM25 unavailable ({e}); hybrid retrieval disabled.")
        bm25_index = None

    # ==================================================================
    # PIPELINE A: BASELINE NAIVE RAG
    # ==================================================================
    logger.info("--- Baseline Naive RAG (top-4 vector) ---")
    baseline_docs = vectorstore.similarity_search(query, k=BASELINE_K)

    # ==================================================================
    # PIPELINE B: ENHANCED RAG (4 enhancements)
    # ==================================================================
    logger.info("--- Enhanced RAG (Decompose + Hybrid + Rerank + Filter) ---")

    # Enhancement #4: Query decomposition for synthesis queries
    decomposed_docs = None
    is_synthesis = _is_synthesis_query(query)
    if is_synthesis:
        logger.info("  Synthesis query detected → decomposing...")
        try:
            decomp_llm = ChatAnthropic(
                model=LLM_MODEL, temperature=0, max_tokens=512
            )
            decomposed_docs = decompose_and_retrieve(
                query, vectorstore, bm25_index, decomp_llm
            )
            time.sleep(API_CALL_DELAY)
        except Exception as e:
            logger.warning(f"  Decomposition pipeline failed: {e}")
            decomposed_docs = None

    if decomposed_docs is not None:
        # Use decomposed multi-query candidates
        fused_docs = decomposed_docs
        logger.info(f"  Using decomposed candidates: {len(fused_docs)} docs.")
    else:
        # Standard single-query hybrid retrieval
        # Enhancement #2: Hybrid BM25 + Vector retrieval with RRF
        vector_candidates = vectorstore.similarity_search(query, k=ENHANCED_VECTOR_K)
        if bm25_index:
            bm25_candidates = bm25_index.search(query, k=ENHANCED_BM25_K)
            fused_docs = reciprocal_rank_fusion(vector_candidates, bm25_candidates)
            logger.info(
                f"  RRF fused {len(vector_candidates)} vector + {len(bm25_candidates)} BM25 "
                f"→ {len(fused_docs)} unique candidates."
            )
        else:
            fused_docs = vector_candidates

    # Enhancement #3: Metadata filtering (optional faceted retrieval)
    if year_filter or source_type_filter:
        fused_docs = apply_metadata_filters(fused_docs, year_filter, source_type_filter)
        logger.info(
            f"  Metadata filter (year={year_filter}, type={source_type_filter}) "
            f"→ {len(fused_docs)} candidates remain."
        )

    # Enhancement #1: Cross-Encoder reranking
    # Use more chunks for synthesis queries so both sides of a comparison fit
    top_k = FINAL_TOP_K + 2 if is_synthesis else FINAL_TOP_K
    logger.info(f"  Cross-Encoder reranking (top_k={top_k})...")
    try:
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        pairs = [[query, doc.page_content] for doc in fused_docs[:20]]
        ce_scores = cross_encoder.predict(pairs)
        scored = sorted(zip(ce_scores, fused_docs[:20]), key=lambda x: x[0], reverse=True)
        enhanced_docs = [doc for _, doc in scored[:top_k]]
    except Exception as e:
        logger.error(f"  Cross-encoder failed: {e}. Falling back to RRF top-k.")
        enhanced_docs = fused_docs[:top_k]

    logger.info(f"  Final enhanced set: {len(enhanced_docs)} chunks.")

    # ==================================================================
    # GENERATION
    # ==================================================================
    logger.info("Generating responses via Claude...")
    try:
        llm = ChatAnthropic(
            model=LLM_MODEL, temperature=0, max_tokens=LLM_MAX_TOKENS
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        sys.exit(1)

    chain = RAG_PROMPT | llm | StrOutputParser()

    logger.info("  → Baseline response...")
    baseline_response = _invoke_chain(chain, format_docs(baseline_docs), query)

    time.sleep(API_CALL_DELAY)  # proactive rate-limit throttle

    logger.info("  → Enhanced response...")
    enhanced_response = _invoke_chain(chain, format_docs(enhanced_docs), query)

    # ==================================================================
    # BUILD RESULT OBJECT
    # ==================================================================
    timestamp = datetime.now().isoformat()
    result = {
        "timestamp": timestamp,
        "query": query,
        "prompt_version": PROMPT_VERSION,
        "filters": {
            "year": year_filter,
            "source_type": source_type_filter,
        },
        "baseline_rag": {
            "retrieved_chunk_ids": [
                d.metadata.get("chunk_id", "UNKNOWN_ID") for d in baseline_docs
            ],
            "retrieved_chunks": [_doc_to_dict(d) for d in baseline_docs],
            "llm_output": baseline_response,
        },
        "enhanced_rag": {
            "enhancements": [
                "query_decomposition_multihop",
                "BM25_hybrid_retrieval_RRF",
                "cross_encoder_reranking",
                "metadata_filtering",
            ],
            "query_decomposed": is_synthesis,
            "retrieved_chunk_ids": [
                d.metadata.get("chunk_id", "UNKNOWN_ID") for d in enhanced_docs
            ],
            "retrieved_chunks": [_doc_to_dict(d) for d in enhanced_docs],
            "llm_output": enhanced_response,
        },
    }

    # Persist to run log + session cache
    _append_run_log(result)
    _cache_session(result)
    logger.info(f"Session cached and logged (timestamp: {timestamp}).")

    # Print comparison
    print("\n" + "=" * 70)
    print("BASELINE RAG OUTPUT (Top-4 Vector Search Only)")
    print("=" * 70)
    print(baseline_response)
    print("\n" + "=" * 70)
    print("ENHANCED RAG OUTPUT (Hybrid BM25+Vector → Cross-Encoder Reranking)")
    print("=" * 70)
    print(enhanced_response)

    return result


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Energy Sector RAG — Baseline vs. Enhanced comparison"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="According to the provided texts, what is the exact financial cost "
                "in USD of deploying GAIA across the entire Texas ERCOT power grid?",
        help="Research question to answer.",
    )
    parser.add_argument("--year", default=None, help="Filter by publication year.")
    parser.add_argument("--source-type", default=None, help="Filter by source type.")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay cached session (no API key needed).",
    )
    args = parser.parse_args()

    result = run_comparative_rag(
        query=args.query,
        year_filter=args.year,
        source_type_filter=args.source_type,
        replay=args.replay,
    )

    if args.replay:
        print("\n" + "=" * 70)
        print(f"[REPLAY] Cached result from {result['timestamp']}")
        print("=" * 70)
        print("\nBASELINE:")
        print(result["baseline_rag"]["llm_output"])
        print("\nENHANCED:")
        print(result["enhanced_rag"]["llm_output"])


if __name__ == "__main__":
    main()
