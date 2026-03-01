# src/app/app.py
# ============================================================================
# Personal Research Portal — Streamlit UI (Phase 3)
#
# Pages:
#   1. Search & Ask       — query with full enhanced pipeline, inline citations
#   2. Corpus Browser     — data manifest, source metadata, chunk stats
#   3. Research Threads   — saved query sessions, export
#   4. Artifact Generator — evidence table + synthesis memo
#   5. Evaluation         — 20-query results, metrics, trust behavior dashboard
#
# Supports DEMO_MODE: reads from session_cache.json, no API key required.
# ============================================================================

import os
import sys
import json
import csv
import io
import pickle
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Resolve paths relative to project root (app lives in src/app/)
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
THREADS_DIR = PROJECT_ROOT / "outputs" / "threads"
ARTIFACTS_DIR = PROJECT_ROOT / "outputs" / "artifacts"
FAISS_INDEX_PATH = DATA_DIR / "processed" / "faiss_index"
BM25_CORPUS_PATH = DATA_DIR / "processed" / "bm25_corpus.pkl"
MANIFEST_PATH = DATA_DIR / "data_manifest.csv"
SESSION_CACHE_PATH = LOGS_DIR / "session_cache.json"
EVAL_RESULTS_PATH = LOGS_DIR / "eval_results.json"
RUN_LOGS_PATH = LOGS_DIR / "run_logs.json"

DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Model & prompt config
# ---------------------------------------------------------------------------
LLM_MODEL = "claude-haiku-4-5-20251001"
PROMPT_VERSION = "v3.1-hybrid-decompose-citations"

# ---------------------------------------------------------------------------
# Lazy imports (only when not in demo mode)
# ---------------------------------------------------------------------------
def _load_langchain():
    """Import heavy ML modules only when needed."""
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from sentence_transformers import CrossEncoder
    return FAISS, HuggingFaceEmbeddings, ChatAnthropic, ChatPromptTemplate, StrOutputParser, CrossEncoder


# ============================================================================
# PROMPTS
# ============================================================================
ANSWER_PROMPT_TEMPLATE = """You are an expert AI research assistant for the energy sector.
Answer the user's query using ONLY the provided context chunks.

RULES:
1. If the context does not contain sufficient evidence, state exactly:
   "I cannot find evidence for this claim in the provided context."
   Then suggest a more specific query the user could try.
2. If the context contains CONFLICTING evidence across sources, flag it:
   "Conflicting evidence found:" and cite both sides.
3. Every factual claim MUST be cited with its Chunk ID.
4. Format your answer as flowing prose with inline citations like (chunk_id).
5. End your answer with a ## Sources section listing each cited chunk ID and a one-line description.

Context:
{context}"""

EVIDENCE_TABLE_TEMPLATE = """You are an expert AI research assistant.
Using ONLY the provided context chunks, generate a structured evidence table.

Output a Markdown table with exactly these columns:
| Claim / Information | Supporting Text Snippet | Chunk ID | Confidence | Notes |

Rules:
- Confidence = High / Medium / Low based on how directly the snippet supports the claim.
- Notes = any caveats, conflicts, or missing evidence.
- If context is insufficient for a row, write "Insufficient evidence" in the Notes column.
- Do NOT invent claims or snippets not present in the context.
- Include 4-8 rows covering the most important evidence for the query.

Context:
{context}"""

SYNTHESIS_MEMO_TEMPLATE = """You are an expert AI research analyst for the energy sector.
Using ONLY the provided context chunks, write a synthesis memo of 800-1200 words.

FORMAT:
## Synthesis Memo: [topic derived from query]
**Date:** [today]
**Sources consulted:** [list unique source_ids from the chunks]

### Background
Brief framing of the research question (2-3 sentences).

### Key Findings
Discuss the main evidence, organized thematically. Every claim must include an inline
citation in the form (chunk_id). Highlight where sources agree and where they diverge.

### Gaps and Limitations
What questions remain unanswered? What evidence is missing or weak?

### Conclusion
2-3 sentence summary of the overall state of evidence.

### References
List each cited chunk_id with its source_id and section.

RULES:
- Do NOT invent claims or citations.
- If evidence is insufficient, explicitly state what is missing.

Context:
{context}"""


# ============================================================================
# HELPERS
# ============================================================================
def format_docs(docs) -> str:
    parts = []
    for doc in docs:
        cid = doc.metadata.get("chunk_id", "UNKNOWN_ID")
        src = doc.metadata.get("source_id", "")
        yr = doc.metadata.get("year", "")
        sec = doc.metadata.get("Section", "")
        header = f"--- Chunk ID: [{cid}] | Source: {src} | Year: {yr} | Section: {sec} ---"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def format_docs_from_dicts(chunk_dicts: list) -> str:
    """Format chunk dicts (from session cache) the same way as LangChain docs."""
    parts = []
    for c in chunk_dicts:
        meta = c.get("metadata", {})
        cid = meta.get("chunk_id", "UNKNOWN_ID")
        src = meta.get("source_id", "")
        yr = meta.get("year", "")
        sec = meta.get("Section", "")
        header = f"--- Chunk ID: [{cid}] | Source: {src} | Year: {yr} | Section: {sec} ---"
        parts.append(f"{header}\n{c.get('page_content', '')}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Resource loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading vector index...")
def load_vectorstore():
    FAISS, HuggingFaceEmbeddings, *_ = _load_langchain()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(
        str(FAISS_INDEX_PATH), embeddings, allow_dangerous_deserialization=True
    )


@st.cache_resource(show_spinner="Loading cross-encoder...")
def load_cross_encoder():
    *_, CrossEncoder = _load_langchain()
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource(show_spinner="Loading BM25 index...")
def load_bm25():
    from rank_bm25 import BM25Okapi
    with open(BM25_CORPUS_PATH, "rb") as f:
        corpus = pickle.load(f)
    tokenized = [entry["text"].lower().split() for entry in corpus]
    return BM25Okapi(tokenized), corpus


def get_llm():
    _, _, ChatAnthropic, *_ = _load_langchain()
    return ChatAnthropic(model=LLM_MODEL, temperature=0, max_tokens=1024)


# ---------------------------------------------------------------------------
# Pipeline runner (uses rag.py for live, session_cache for demo)
# ---------------------------------------------------------------------------
def run_query_live(query: str, vectorstore, cross_encoder, bm25_data, llm):
    """Full enhanced pipeline — live API calls."""
    from src.rag.rag import run_comparative_rag
    result = run_comparative_rag(query=query)
    return result


def load_session_cache() -> list:
    if SESSION_CACHE_PATH.exists():
        with open(SESSION_CACHE_PATH) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def find_cached(query: str, cache: list) -> dict | None:
    for entry in reversed(cache):
        if entry.get("query", "").strip().lower() == query.strip().lower():
            return entry
    return None


# ---------------------------------------------------------------------------
# Thread management
# ---------------------------------------------------------------------------
def save_thread(query, answer, chunks_info, artifact=""):
    THREADS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    thread = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "prompt_version": PROMPT_VERSION,
        "retrieved_chunks": chunks_info,
        "answer": answer,
        "artifact": artifact,
    }
    path = THREADS_DIR / f"thread_{ts}.json"
    with open(path, "w") as f:
        json.dump(thread, f, indent=2)
    return path


def load_threads():
    if not THREADS_DIR.exists():
        return []
    files = sorted(THREADS_DIR.glob("thread_*.json"), reverse=True)
    threads = []
    for f in files:
        with open(f) as fp:
            try:
                threads.append((f.name, json.load(fp)))
            except Exception:
                pass
    return threads


def export_markdown(query, answer, chunks_info, artifact=""):
    lines = [
        "# Research Thread Export",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Query:** {query}",
        f"**Prompt Version:** {PROMPT_VERSION}",
        "", "---", "",
        "## Answer", answer, "",
    ]
    if artifact:
        lines += ["---", "", "## Research Artifact", artifact, ""]
    lines += ["---", "", "## Retrieved Chunks"]
    for c in chunks_info:
        cid = c.get("chunk_id", "UNKNOWN")
        sid = c.get("source_id", "")
        sec = c.get("section", "")
        lines.append(f"\n### [{cid}] — {sid} | {sec}")
        lines.append(c.get("snippet", ""))
    return "\n".join(lines)


def export_csv(query, answer, chunks_info):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["query", "chunk_id", "source_id", "section", "snippet", "answer"])
    for c in chunks_info:
        writer.writerow([
            query,
            c.get("chunk_id", ""),
            c.get("source_id", ""),
            c.get("section", ""),
            c.get("snippet", "")[:300],
            answer[:500],
        ])
    return output.getvalue()


# ============================================================================
# PAGE: SEARCH & ASK
# ============================================================================
def page_search():
    st.title("🔍 Search & Ask")
    st.caption("Ask a research question. Every claim is precise, traceable, and cited. Full transparency is ensured.")

    # --- Query form ---
    with st.form("query_form"):
        query = st.text_area(
            "Research Question",
            placeholder="e.g. What are the key challenges of deploying LLMs in power grid management?",
            height=100,
        )
        col1, col2 = st.columns(2)
        pipeline = col1.selectbox(
            "Pipeline",
            ["Enhanced (Decompose + Hybrid + Rerank)", "Baseline (Vector Only)"],
        )
        gen_artifact = col2.selectbox(
            "Generate Artifact",
            ["None", "Evidence Table", "Synthesis Memo"],
        )
        submitted = st.form_submit_button("Ask", use_container_width=True)

    if not submitted or not query.strip():
        # Show sample queries when idle
        st.markdown("**Sample queries to try:**")
        samples = [
            "What specific tasks does the GAIA LLM address in power dispatch?",
            "Compare EnergyGPT deployment challenges with Sarwar's computational cost limitations.",
            "Does the corpus contain evidence that LLMs can actuate robotic arms?",
        ]
        for sq in samples:
            if st.button(sq, key=f"sample_{sq[:30]}"):
                st.session_state["auto_query"] = sq
                st.rerun()
        return

    use_enhanced = "Enhanced" in pipeline

    # --- Determine if demo mode or live ---
    if DEMO_MODE or not os.environ.get("ANTHROPIC_API_KEY"):
        # DEMO / REPLAY MODE
        cache = load_session_cache()
        cached = find_cached(query, cache)
        if cached:
            st.info("📼 **Demo mode:** Showing cached results (no API key needed).")
            rag_key = "enhanced_rag" if use_enhanced else "baseline_rag"
            answer = cached[rag_key]["llm_output"]
            chunk_ids = cached[rag_key]["retrieved_chunk_ids"]
            chunks_data = cached[rag_key].get("retrieved_chunks", [])
            chunks_info = [
                {
                    "chunk_id": c.get("metadata", {}).get("chunk_id", ""),
                    "source_id": c.get("metadata", {}).get("source_id", ""),
                    "section": c.get("metadata", {}).get("Section", ""),
                    "snippet": c.get("page_content", "")[:300],
                }
                for c in chunks_data
            ]
            was_decomposed = cached.get("enhanced_rag", {}).get("query_decomposed", False)
        else:
            st.warning(
                "📼 **Demo mode:** No cached result found for this exact query. "
                "Try one of the 20 evaluation queries, or provide an API key for live queries."
            )
            return
    else:
        # LIVE MODE
        with st.spinner("Retrieving and generating answer..."):
            try:
                result = run_query_live(query, None, None, None, None)
                rag_key = "enhanced_rag" if use_enhanced else "baseline_rag"
                answer = result[rag_key]["llm_output"]
                chunk_ids = result[rag_key]["retrieved_chunk_ids"]
                chunks_data = result[rag_key].get("retrieved_chunks", [])
                chunks_info = [
                    {
                        "chunk_id": c.get("metadata", {}).get("chunk_id", ""),
                        "source_id": c.get("metadata", {}).get("source_id", ""),
                        "section": c.get("metadata", {}).get("Section", ""),
                        "snippet": c.get("page_content", "")[:300],
                    }
                    for c in chunks_data
                ]
                was_decomposed = result.get("enhanced_rag", {}).get("query_decomposed", False)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                return

    # --- Display answer ---
    st.subheader("📝 Answer")
    if was_decomposed and use_enhanced:
        st.caption("🔀 Query was decomposed into sub-queries for multi-source retrieval.")

    # Detect abstention
    if "cannot find evidence" in answer.lower():
        st.warning("⚠️ The system could not find sufficient evidence in the corpus for this query.")
    if "conflicting evidence" in answer.lower():
        st.info("⚖️ Conflicting evidence was detected across sources.")

    st.markdown(answer)

    # --- Artifact generation ---
    artifact = ""
    if gen_artifact != "None" and not DEMO_MODE and os.environ.get("ANTHROPIC_API_KEY"):
        with st.spinner(f"Generating {gen_artifact}..."):
            try:
                llm = get_llm()
                _, _, _, ChatPromptTemplate, StrOutputParser, _ = _load_langchain()

                context = format_docs_from_dicts(chunks_data)

                if gen_artifact == "Evidence Table":
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", EVIDENCE_TABLE_TEMPLATE),
                        ("human", "Query: {question}"),
                    ])
                else:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", SYNTHESIS_MEMO_TEMPLATE),
                        ("human", "Query: {question}"),
                    ])

                chain = prompt | llm | StrOutputParser()
                artifact = chain.invoke({"context": context, "question": query})
                time.sleep(2)  # rate limit
            except Exception as e:
                st.error(f"Artifact generation failed: {e}")

    if artifact:
        st.subheader(f"📊 {gen_artifact}")
        st.markdown(artifact)

        # Save artifact to file
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_path = ARTIFACTS_DIR / f"{gen_artifact.lower().replace(' ', '_')}_{ts}.md"
        with open(artifact_path, "w") as f:
            f.write(f"# {gen_artifact}\n\n**Query:** {query}\n**Date:** {datetime.now().isoformat()}\n\n{artifact}")
        st.success(f"Artifact saved to `{artifact_path.relative_to(PROJECT_ROOT)}`")

    # --- Retrieved chunks ---
    with st.expander(f"📚 Retrieved Chunks ({len(chunks_info)})", expanded=False):
        for c in chunks_info:
            st.markdown(f"**[{c['chunk_id']}]** — `{c['source_id']}` | *{c['section']}*")
            st.caption(c["snippet"] + "..." if len(c["snippet"]) >= 295 else c["snippet"])
            st.divider()

    # --- Save thread & export ---
    thread_path = save_thread(query, answer, chunks_info, artifact)
    st.success(f"Thread saved: `{thread_path.name}`")

    col_a, col_b = st.columns(2)
    md_export = export_markdown(query, answer, chunks_info, artifact)
    col_a.download_button(
        "⬇️ Export Markdown",
        data=md_export,
        file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    csv_export = export_csv(query, answer, chunks_info)
    col_b.download_button(
        "⬇️ Export CSV",
        data=csv_export,
        file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================================
# PAGE: CORPUS BROWSER
# ============================================================================
def page_corpus():
    st.title("📁 Corpus Browser")
    st.caption("Browse the data manifest and source metadata.")

    if not MANIFEST_PATH.exists():
        st.error(f"Data manifest not found at `{MANIFEST_PATH}`.")
        return

    df = pd.read_csv(MANIFEST_PATH)
    df.fillna("", inplace=True)

    # Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sources", len(df))
    c2.metric("Year Range", f"{df['year'].min()} – {df['year'].max()}" if "year" in df.columns else "N/A")
    if "source_type" in df.columns:
        types = df["source_type"].value_counts()
        c3.metric("Source Types", ", ".join(f"{k} ({v})" for k, v in types.items()))

    st.divider()

    # Chunk stats from ingest log
    ingest_log_path = LOGS_DIR / "ingest_log.json"
    if ingest_log_path.exists():
        with open(ingest_log_path) as f:
            ingest_data = json.load(f)
        ingest_df = pd.DataFrame(ingest_data)
        total_chunks = ingest_df["chunks_created"].sum()
        st.caption(f"**Total chunks in index:** {total_chunks} across {len(ingest_df)} sources")

        # Merge chunk counts into manifest
        if "source_id" in df.columns and "source_id" in ingest_df.columns:
            merged = df.merge(
                ingest_df[["source_id", "headers_found", "chunks_created"]],
                on="source_id", how="left"
            )
        else:
            merged = df
    else:
        merged = df

    # Display columns
    display_cols = [c for c in [
        "source_id", "title", "year", "source_type", "venue",
        "tags", "headers_found", "chunks_created"
    ] if c in merged.columns]

    st.dataframe(merged[display_cols], use_container_width=True, hide_index=True)

    # Source detail expander
    st.divider()
    st.subheader("Source Details")
    for _, row in df.iterrows():
        with st.expander(f"📄 {row.get('source_id', 'Unknown')} — {row.get('title', '')[:60]}"):
            for col in df.columns:
                val = row[col]
                if val:
                    st.markdown(f"**{col}:** {val}")


# ============================================================================
# PAGE: RESEARCH THREADS
# ============================================================================
def page_threads():
    st.title("🗂️ Research Threads")
    st.caption("All saved query sessions with evidence and answers.")

    threads = load_threads()
    if not threads:
        st.info("No threads saved yet. Run a query on the Search & Ask page.")
        return

    st.metric("Saved Threads", len(threads))
    st.divider()

    for fname, thread in threads:
        ts = thread.get("timestamp", "")[:16].replace("T", " ")
        q = thread.get("query", "(no query)")
        with st.expander(f"🕐 {ts} — {q[:80]}"):
            st.markdown(f"**Query:** {q}")
            st.markdown(f"**Prompt version:** `{thread.get('prompt_version', '')}`")

            st.markdown("### Answer")
            st.markdown(thread.get("answer", ""))

            if thread.get("artifact"):
                st.markdown("### Research Artifact")
                st.markdown(thread["artifact"])

            st.markdown("### Retrieved Chunks")
            for chunk in thread.get("retrieved_chunks", []):
                st.caption(
                    f"`{chunk.get('chunk_id', '')}` — {chunk.get('source_id', '')} "
                    f"| {chunk.get('section', '')}\n\n"
                    f"*{chunk.get('snippet', '')}...*"
                )

            md = export_markdown(
                q, thread.get("answer", ""),
                thread.get("retrieved_chunks", []),
                thread.get("artifact", ""),
            )
            st.download_button(
                "⬇️ Export Thread",
                data=md,
                file_name=fname.replace(".json", ".md"),
                mime="text/markdown",
                key=f"dl_{fname}",
            )


# ============================================================================
# PAGE: ARTIFACT GENERATOR (standalone)
# ============================================================================
def page_artifacts():
    st.title("📊 Artifact Generator")
    st.caption("Generate research artifacts from cached sessions or live queries.")

    cache = load_session_cache()
    if not cache:
        st.warning("No cached sessions found. Run some queries first.")
        return

    # Let user pick from cached queries
    query_options = [e["query"][:100] for e in cache]
    selected_idx = st.selectbox("Select a cached query", range(len(query_options)),
                                format_func=lambda i: query_options[i])

    entry = cache[selected_idx]
    chunks_data = entry.get("enhanced_rag", {}).get("retrieved_chunks", [])

    st.markdown(f"**Full query:** {entry['query']}")
    st.markdown(f"**Chunks available:** {len(chunks_data)}")

    artifact_type = st.selectbox("Artifact Type", ["Evidence Table", "Synthesis Memo"])

    if DEMO_MODE or not os.environ.get("ANTHROPIC_API_KEY"):
        st.info("📼 Artifact generation requires an API key. Demo mode can only display cached answers.")
        st.markdown("### Cached Enhanced Answer")
        st.markdown(entry.get("enhanced_rag", {}).get("llm_output", "No output available"))
        return

    if st.button("Generate Artifact", use_container_width=True):
        with st.spinner(f"Generating {artifact_type}..."):
            try:
                llm = get_llm()
                _, _, _, ChatPromptTemplate, StrOutputParser, _ = _load_langchain()
                context = format_docs_from_dicts(chunks_data)

                if artifact_type == "Evidence Table":
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", EVIDENCE_TABLE_TEMPLATE),
                        ("human", "Query: {question}"),
                    ])
                else:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", SYNTHESIS_MEMO_TEMPLATE),
                        ("human", "Query: {question}"),
                    ])

                chain = prompt | llm | StrOutputParser()
                artifact = chain.invoke({"context": context, "question": entry["query"]})
            except Exception as e:
                st.error(f"Generation failed: {e}")
                return

        st.markdown(f"### {artifact_type}")
        st.markdown(artifact)

        # Save
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        atype = artifact_type.lower().replace(" ", "_")
        path = ARTIFACTS_DIR / f"{atype}_{ts}.md"
        with open(path, "w") as f:
            f.write(f"# {artifact_type}\n\n**Query:** {entry['query']}\n**Date:** {datetime.now().isoformat()}\n\n{artifact}")
        st.success(f"Saved to `{path.relative_to(PROJECT_ROOT)}`")

        st.download_button(
            f"⬇️ Download {artifact_type}",
            data=artifact,
            file_name=f"{atype}_{ts}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ============================================================================
# PAGE: EVALUATION
# ============================================================================
ABSTENTION_PHRASES = [
    "cannot find evidence", "not in the provided context", "not supported by",
    "not mentioned in", "does not contain", "no evidence",
    "not available in the provided", "not found in the provided",
]

def _detected_abstention(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in ABSTENTION_PHRASES)


def page_evaluation():
    st.title("📈 Evaluation Dashboard")
    st.caption("Results from the 20-query evaluation set. Loaded from `logs/eval_results.json` and `logs/session_cache.json`.")

    # --- Try eval_results.json first, then fall back to session cache ---
    eval_data = None
    if EVAL_RESULTS_PATH.exists():
        with open(EVAL_RESULTS_PATH) as f:
            try:
                eval_data = json.load(f)
            except json.JSONDecodeError:
                pass

    if eval_data:
        summary = eval_data.get("summary", {})
        results = eval_data.get("results", [])
    else:
        # Fall back to run_logs.json
        if not RUN_LOGS_PATH.exists():
            st.error("No evaluation data found. Run `make eval` first.")
            return
        with open(RUN_LOGS_PATH) as f:
            raw_logs = json.load(f)
        results = _parse_legacy_logs(raw_logs)
        summary = {}

    if not results:
        st.warning("No evaluation results found.")
        return

    # --- Summary metrics ---
    st.subheader("Summary Metrics")

    total = len(results)
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    direct_count = categories.get("direct", 0)
    synth_count = categories.get("synthesis", 0)
    edge_count = categories.get("edge_case", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Queries", total)
    c2.metric("Direct", direct_count)
    c3.metric("Synthesis", synth_count)
    c4.metric("Edge Case", edge_count)

    # Trust behavior
    st.divider()
    st.subheader("Trust Behavior")

    edge_results = [r for r in results if r.get("category") == "edge_case"]
    base_edge_abstain = sum(1 for r in edge_results if r.get("baseline_abstained", False))
    enh_edge_abstain = sum(1 for r in edge_results if r.get("enhanced_abstained", False))

    direct_results = [r for r in results if r.get("category") == "direct"]
    base_direct_abstain = sum(1 for r in direct_results if r.get("baseline_abstained", False))
    enh_direct_abstain = sum(1 for r in direct_results if r.get("enhanced_abstained", False))

    synth_results = [r for r in results if r.get("category") == "synthesis"]
    base_synth_abstain = sum(1 for r in synth_results if r.get("baseline_abstained", False))
    enh_synth_abstain = sum(1 for r in synth_results if r.get("enhanced_abstained", False))

    trust_data = {
        "Category": ["Direct", "Synthesis", "Edge Case"],
        "Count": [direct_count, synth_count, edge_count],
        "Baseline Abstentions": [base_direct_abstain, base_synth_abstain, base_edge_abstain],
        "Enhanced Abstentions": [enh_direct_abstain, enh_synth_abstain, enh_edge_abstain],
    }
    trust_df = pd.DataFrame(trust_data)
    st.dataframe(trust_df, use_container_width=True, hide_index=True)

    if edge_count > 0:
        rate = enh_edge_abstain / edge_count
        if rate == 1.0:
            st.success(f"✅ Edge-case abstention rate: {rate:.0%} — all adversarial queries correctly refused.")
        else:
            st.warning(f"⚠️ Edge-case abstention rate: {rate:.0%}")

    # Enhanced vs baseline improvement
    base_total_abstain = base_direct_abstain + base_synth_abstain
    enh_total_abstain = enh_direct_abstain + enh_synth_abstain
    non_edge_total = direct_count + synth_count
    if non_edge_total > 0:
        base_answer_rate = (non_edge_total - base_total_abstain) / non_edge_total
        enh_answer_rate = (non_edge_total - enh_total_abstain) / non_edge_total
        st.caption(
            f"**Answer rate (non-edge queries):** Baseline {base_answer_rate:.0%} → "
            f"Enhanced {enh_answer_rate:.0%} "
            f"({'↑ improved' if enh_answer_rate > base_answer_rate else 'same'})"
        )

    # --- Per-query detail ---
    st.divider()
    st.subheader("Per-Query Results")

    filter_cat = st.radio(
        "Filter by category",
        ["All", "Direct", "Synthesis", "Edge Case"],
        horizontal=True,
    )

    filtered = results
    if filter_cat != "All":
        key = filter_cat.lower().replace(" ", "_")
        filtered = [r for r in results if r.get("category") == key]

    for r in filtered:
        qid = r.get("query_id", "?")
        cat = r.get("category", "?")
        base_abs = r.get("baseline_abstained", False)
        enh_abs = r.get("enhanced_abstained", False)

        if cat == "edge_case":
            badge = "✅" if enh_abs else "❌"
        else:
            badge = "🟢" if not enh_abs else "🔴"

        q_short = r.get("query", "")[:80]

        with st.expander(f"{badge} [{qid}] {q_short}"):
            st.caption(f"Category: **{cat}** | Baseline abstained: {base_abs} | Enhanced abstained: {enh_abs}")

            # Enhanced chunk sources
            chunks = r.get("enhanced_chunk_ids", [])
            sources = sorted(set(c.rsplit("_chunk_", 1)[0] for c in chunks if "_chunk_" in c))
            st.caption(f"Enhanced sources: {', '.join(sources)} ({len(chunks)} chunks)")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Baseline**")
                b_out = r.get("baseline_output", "")
                if b_out:
                    st.markdown(b_out[:1500] + ("..." if len(b_out) > 1500 else ""))
                else:
                    st.caption("No baseline output available.")
            with col2:
                st.markdown("**Enhanced**")
                e_out = r.get("enhanced_output", "")
                if e_out:
                    st.markdown(e_out[:1500] + ("..." if len(e_out) > 1500 else ""))
                else:
                    st.caption("No enhanced output available.")

    # --- Export ---
    st.divider()
    md_lines = [
        "# Evaluation Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Queries:** {total} (Direct: {direct_count}, Synthesis: {synth_count}, Edge: {edge_count})",
        f"**Edge-case abstention rate (enhanced):** {enh_edge_abstain}/{edge_count}",
        f"**Non-edge answer rate:** Baseline {non_edge_total - base_total_abstain}/{non_edge_total} "
        f"→ Enhanced {non_edge_total - enh_total_abstain}/{non_edge_total}",
        "", "---", "",
    ]
    for r in results:
        qid = r.get("query_id", "?")
        cat = r.get("category", "?")
        md_lines += [
            f"## [{qid}] ({cat})",
            f"**Query:** {r.get('query', '')}",
            f"**Enhanced abstained:** {r.get('enhanced_abstained', '')}",
            f"**Enhanced chunks:** {', '.join(r.get('enhanced_chunk_ids', []))}",
            "", r.get("enhanced_output", "")[:2000], "", "---", "",
        ]
    md_export = "\n".join(md_lines)

    st.download_button(
        "⬇️ Export Evaluation Report (Markdown)",
        data=md_export,
        file_name=f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
    )


def _parse_legacy_logs(logs):
    """Parse old-format run_logs.json into eval-compatible format."""
    results = []
    seen = {}
    for entry in logs:
        q = entry.get("query", "")
        if "baseline_rag" in entry and "enhanced_rag" in entry:
            base_out = entry["baseline_rag"].get("llm_output", "")
            enh_out = entry["enhanced_rag"].get("llm_output", "")
            r = {
                "query_id": f"L{len(seen)+1:02d}",
                "category": "unknown",
                "query": q,
                "status": "OK",
                "baseline_output": base_out,
                "enhanced_output": enh_out,
                "baseline_abstained": _detected_abstention(base_out),
                "enhanced_abstained": _detected_abstention(enh_out),
                "baseline_chunk_ids": entry["baseline_rag"].get("retrieved_chunk_ids", []),
                "enhanced_chunk_ids": entry["enhanced_rag"].get("retrieved_chunk_ids", []),
                "timestamp": entry.get("timestamp", ""),
            }
            seen[q] = r
        elif q not in seen:
            out = entry.get("llm_output", "")
            r = {
                "query_id": f"L{len(seen)+1:02d}",
                "category": "unknown",
                "query": q,
                "status": "OK",
                "baseline_output": "",
                "enhanced_output": out,
                "baseline_abstained": False,
                "enhanced_abstained": _detected_abstention(out),
                "baseline_chunk_ids": [],
                "enhanced_chunk_ids": entry.get("retrieved_chunk_ids", []),
                "timestamp": entry.get("timestamp", ""),
            }
            seen[q] = r
    return list(seen.values())


# ============================================================================
# MAIN
# ============================================================================
def main():
    st.set_page_config(
        page_title="Energy Sector Research Portal",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Sidebar ---
    with st.sidebar:
        st.title("🔬 Research Portal")
        st.caption("Energy Sector · LLM Research")

        if DEMO_MODE:
            st.info("📼 **Demo Mode** — reading from cached sessions.")
        else:
            st.divider()
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                value=os.environ.get("ANTHROPIC_API_KEY", ""),
                help="Enter your key or set ANTHROPIC_API_KEY in .env",
            )
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key

        st.divider()
        page = st.radio(
            "Navigate",
            [
                "🔍 Search & Ask",
                "📁 Corpus Browser",
                "🗂️ Research Threads",
                "📊 Artifact Generator",
                "📈 Evaluation",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption(f"Prompt: `{PROMPT_VERSION}`")
        st.caption(f"Model: `{LLM_MODEL}`")
        st.caption(f"Mode: `{'DEMO' if DEMO_MODE else 'LIVE'}`")

    # --- Guard: need at least indices for Search ---
    if page == "🔍 Search & Ask" and not DEMO_MODE:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            cache = load_session_cache()
            if cache:
                st.info(
                    "No API key provided. Showing cached results in demo mode. "
                    "Enter your key in the sidebar for live queries."
                )
            else:
                st.warning("⚠️ Enter your Anthropic API key in the sidebar, or run with `make demo-replay`.")

    # --- Route ---
    if page == "🔍 Search & Ask":
        page_search()
    elif page == "📁 Corpus Browser":
        page_corpus()
    elif page == "🗂️ Research Threads":
        page_threads()
    elif page == "📊 Artifact Generator":
        page_artifacts()
    elif page == "📈 Evaluation":
        page_evaluation()


if __name__ == "__main__":
    main()
