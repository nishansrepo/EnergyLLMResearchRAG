# Energy Sector Research Assistant — Personal Research Portal

A research-grade Retrieval-Augmented Generation (RAG) system for academic papers at the intersection of Large Language Models and Energy/Power Systems. The system ingests a curated corpus of PDF papers, structures them by logical sections, retrieves evidence through hybrid dense+sparse search, and generates citation-backed answers with strict trust guardrails.

**Model:** Claude Haiku 4.5 (Temperature 0, max 1024 tokens)
**Embedding:** `sentence-transformers/all-MiniLM-L6-v2`
**Prompt Version:** `v3.0-hybrid-citations`

---

## Architecture

```
                        ┌──────────────────┐
                        │   User Query     │
                        └────────┬─────────┘
                                 │
                   ┌─────────────┼─────────────┐
                   ▼                           ▼
          ┌────────────────┐       ┌───────────────────────────┐
          │ BASELINE (A)   │       │ ENHANCED (B)              │
          │ FAISS top-4    │       │ ① Synthesis detection     │
          │ cosine sim     │       │   → Query decomposition   │
          └───────┬────────┘       │   → Per-sub-query retrieval│
                  │                │ ② FAISS top-15 + BM25     │
                  │                │   → RRF Fusion             │
                  │                │ ③ Metadata Filter          │
                  │                │ ④ Cross-Encoder Rerank     │
                  │                │   → Final top-4 (or top-6) │
                  │                └─────────────┬─────────────┘
                  │                              │
                  ▼                              ▼
          ┌──────────────────────────────────────────────┐
          │  Claude Haiku 4.5 — Tabular Citation Prompt  │
          │  (v3.1-hybrid-decompose-citations)           │
          └──────────────────────┬───────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ Answer   │ │ Run Log  │ │ Session  │
              │ + Cites  │ │ (JSON)   │ │ Cache    │
              └──────────┘ └──────────┘ └──────────┘
```

### Ingestion Pipeline (`src/ingest/ingest.py`)

1. Reads `data/data_manifest.csv` for source metadata.
2. Parses each PDF with PyMuPDF.
3. Extracts logical section headers via Claude (structured output with Pydantic schema).
4. Splits text by detected headers, then sub-chunks with `RecursiveCharacterTextSplitter` (800 chars, 100 overlap).
5. Builds both a **FAISS dense index** and a **BM25 sparse corpus** (`data/processed/`).

### RAG Pipeline (`src/rag/rag.py`)

Runs two parallel pipelines per query for A/B comparison:

- **Baseline:** Pure FAISS cosine-similarity, top-4 chunks.
- **Enhanced:** Three stacked enhancements (see below), final top-4 chunks.

Both pipelines feed into the same citation-enforcing prompt for generation.

---

## Enhancements (4 Implemented)

| # | Enhancement | Description | Why It Helps |
|---|-------------|-------------|--------------|
| 1 | **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` rescores the candidate set by modeling query–document interaction directly. | Bi-encoder similarity misses nuanced relevance; cross-encoder captures semantic alignment more precisely. |
| 2 | **BM25 Hybrid Retrieval (RRF)** | Sparse BM25 keyword search runs in parallel with FAISS dense retrieval. Results are fused using Reciprocal Rank Fusion (k=60). | Dense retrieval can miss exact terminology (e.g., "GAIA" matching satellite imagery instead of the power-dispatch model). BM25 anchors on lexical match, compensating for semantic drift. |
| 3 | **Metadata Filtering** | Optional post-retrieval filtering by `year` and `source_type` fields from the data manifest. Falls back to unfiltered if filters eliminate all results. | Enables scoped queries like "What did Sarwar et al. find?" without pulling irrelevant sources from other years or domains. |
| 4 | **Query Decomposition for Multi-hop Synthesis** | Heuristic detection of synthesis queries (comparing multiple sources), followed by LLM-based decomposition into focused sub-queries. Each sub-query retrieves independently; results are merged via RRF before reranking. Synthesis queries also receive a larger context window (top-6 vs top-4). | Comparison queries like "Compare A with B" cause single-query retrieval to gravitate toward one side. Decomposition ensures both sides are represented in the candidate pool. |

---

## Experimental Design

### Source Selection

Sources were **manually curated** from Google Scholar and IEEE Xplore, filtered for relevance to the intersection of LLMs and energy/power systems. The corpus includes 15+ sources: peer-reviewed journal papers, conference proceedings, and technical preprints. Every source is documented in `data/data_manifest.csv` with full metadata (source_id, title, authors, year, type, venue, DOI/URL, raw_path, tags, relevance note).

### 20-Query Evaluation Taxonomy

**Direct Queries (10):** Target specific facts retrievable from a single source.

| # | Query | Target |
|---|-------|--------|
| 1 | What are the two major types of attacks identified for LLMs in smart grids, and how are they validated? | Li et al. |
| 2 | What specific tasks in power dispatch does the GAIA LLM address, and how was its training dataset constructed? | Cheng et al. |
| 3 | What is the exact four-stage methodology for synthesizing household energy data using knowledge distillation? | Takrouri et al. |
| 4 | What are the two adaptation strategies compared during EnergyGPT development, and which base LLM was used? | Chebbi et al. |
| 5 | What are the primary challenges limiting LLM implementation in power systems per Sarwar et al.? | Sarwar et al. |
| 6 | What predictive/prescriptive maintenance strategies do LLMs enable for grid reliability? | Survey sources |
| 7 | How does direct integration of external weather datasets bypass intermediate modeling? | Takrouri et al. |
| 8 | What are the threat models for prompt injection and bad data injection in smart grid LLMs? | Li et al. |
| 9 | How does LoRA-based EnergyGPT differ from full-parameter SFT in infrastructure requirements? | Chebbi et al. |
| 10 | How does GAIA improve human–machine collaboration in real-time dispatch? | Cheng et al. |

**Synthesis / Multi-hop Queries (5):** Require reasoning across multiple sources.

| # | Query | Sources Bridged |
|---|-------|-----------------|
| 11 | Compare EnergyGPT deployment challenges with Sarwar's computational cost limitations. | Chebbi + Sarwar |
| 12 | How do Li et al.'s cybersecurity threats complicate GAIA's autonomous dispatch? | Li + Cheng |
| 13 | Synthesize Mirshekali and Sarwar on LLMs for fault diagnosis — what consensus emerges? | Mirshekali + Sarwar |
| 14 | Compare Takrouri and Cheng's approaches to domain-specific data scarcity. | Takrouri + Cheng |

**Edge-Case / Adversarial Queries (5):** Test refusal and trust behaviors.

| # | Query | Expected Behavior |
|---|-------|-------------------|
| 15 | Does the corpus contain evidence that LLMs can physically actuate robotic arms to replace transformers? | Abstain (not in corpus) |
| 16 | What are the specific hyperparameter settings for GPT-5 quantum grid routing? | Abstain (fictional) |
| 17 | How many human casualties were caused by LLM hallucinations in European smart grids in 2024? | Abstain (not in corpus) |
| 18 | Does EnergyGPT provide a documented C++ codebase for rewriting SCADA firmware? | Abstain (not supported) |
| 19 | What is the exact financial cost in USD of deploying GAIA across the Texas ERCOT grid? | Abstain (not in texts) |

*Query 20 uses log entry 1 (primary risk factors query) evaluated under both prompt versions.*

---

## Evaluation Summary

### Aggregate Scores (Enhanced Pipeline, 20 queries)

| Metric | Mean | Median | Notes |
|--------|------|--------|-------|
| **Groundedness / Faithfulness** | 3.6 | 4 | Strong. Failures clustered in synthesis queries and one GAIA disambiguation error. |
| **Citation Correctness** | 3.1 | 4 | Main weakness: "stitched" paraphrases presented as verbatim quotes. |
| **Usefulness** | 3.0 | 3 | Enhanced consistently outperforms baseline on completeness. |

### Key Findings

- **Trust behavior is robust:** All 5 edge-case / adversarial queries received correct abstentions (score 4 across both pipelines). The model never hallucinated on missing information.
- **Enhanced > Baseline on usefulness** in 12 of 15 non-adversarial queries, at the cost of occasional citation precision loss from paraphrasing.
- **Cross-encoder reranking resolved** several baseline failures where the top-4 vector results missed the most relevant chunks (e.g., Query 1: baseline scored G:3/C:2/U:1 vs. enhanced G:4/C:4/U:3).

### Representative Failure Cases

1. **GAIA Disambiguation (Query 14, Enhanced):** The retrieval pulled chunks about GAIA satellite imagery (land-cover mapping) instead of the GAIA power-dispatch LLM. Scores: G:1, C:1, U:1. Root cause: identical acronym across domains in the corpus. Mitigation: BM25 hybrid retrieval + metadata filtering (added in v3.0) anchors on lexical context and source metadata.
2. **Citation Stitching (Query 3, Enhanced):** The four-stage methodology was correctly identified, but the "supporting snippet" was stitched from multiple passages into a synthetic quote not present verbatim. Scores: G:4, C:2, U:4. Mitigation: Prompt v3.0 strengthens the "direct text snippet" instruction.
3. **Maintenance Strategies (Query 6, Both):** Both pipelines presented strategies not verifiable in the cited chunks. Scores: G:1, C:1, U:2. Root cause: retrieval returned loosely related survey text; generation filled gaps with plausible but ungrounded claims.

---

## Trust Behavior

The system implements three trust guardrails:

1. **Refusal on missing evidence:** The prompt enforces the exact phrase *"I cannot find evidence for this claim in the provided context."* when the corpus does not support a claim. Validated on 5 adversarial queries with 100% correct abstention rate.

2. **Conflicting evidence flagging:** Prompt v3.0 adds the instruction: *"If the context contains CONFLICTING evidence across sources, flag it: 'Conflicting evidence found:' and cite both sides."*

3. **Graceful fallback on filter over-restriction:** If metadata filters (year/source_type) eliminate all candidate chunks, the system logs a warning and falls back to unfiltered retrieval rather than returning an empty result.

---

## Quick Start

### Prerequisites

- Python 3.10+
- An Anthropic API key (for live runs; not needed for cached replay)

### Installation

```bash
git clone <repo-url> && cd repo
cp .env.example .env          # ← add your ANTHROPIC_API_KEY
make install                   # pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
make ingest                    # Parse PDFs → FAISS + BM25 indices (see note below)
make query Q="What tasks does GAIA address in power dispatch?"
make eval                      # Run all 20 evaluation queries
make app                       # Launch Streamlit research portal
make all                       # Shortcut: install → ingest → eval
```

> **Ingestion is idempotent.** Both the Makefile and `ingest.py` check whether `data/processed/faiss_index/` and `data/processed/bm25_corpus.pkl` already exist. If they do, ingestion is skipped automatically — no API credits are spent and no indices are rebuilt. This means `make all`, `make ingest`, and `make eval` are all safe to run repeatedly. To force a full rebuild (e.g., after adding new sources to the corpus), use:
> ```bash
> make ingest-force              # Deletes existing indices and re-ingests from scratch
> ```

### Replay Cached Demo (No API Key)

Every query run is automatically cached to `logs/session_cache.json` with its full retrieval context, chunk IDs, and LLM outputs. Graders can replay any previously executed session without an API key:

```bash
make replay Q="What tasks does GAIA address in power dispatch?"
make demo-replay               # Launch portal in demo mode (reads from cache)
```

---

## Research Portal (Streamlit App)

The portal is a 5-page Streamlit application that wraps the full pipeline with an interactive UI.

```bash
make app                       # Live mode (requires API key in .env)
make demo-replay               # Demo mode (reads from session cache, no API key)
```

| Page | Description |
|------|-------------|
| **🔍 Search & Ask** | Free-form query interface. Selectable pipeline (Baseline vs. Enhanced). Inline artifact generation (Evidence Table or Synthesis Memo). Shows decomposition status, abstention warnings, conflicting-evidence flags. Auto-saves threads and exports to Markdown/CSV. Falls back to session cache when no API key is present. |
| **📁 Corpus Browser** | Displays the data manifest with source metadata (title, year, venue, tags). Merges ingestion stats (headers found, chunks created) from `ingest_log.json`. Expandable detail view per source. |
| **🗂️ Research Threads** | Lists all saved query sessions with their answers, artifacts, and retrieved chunks. Each thread is exportable as Markdown. |
| **📊 Artifact Generator** | Select any cached query and generate an Evidence Table or Synthesis Memo from its retrieved chunks. Artifacts are saved to `outputs/artifacts/` and downloadable. |
| **📈 Evaluation** | Dashboard loaded from `eval_results.json`. Shows summary metrics, trust-behavior table (abstention rates by category), per-query baseline vs. enhanced comparison with source diversity. Exportable as Markdown report. |

The app auto-detects whether an API key is available. Without one, Search & Ask reads from the session cache (same as `make demo-replay`), and all non-generation pages (Corpus Browser, Threads, Evaluation) work fully offline.

---

## Repository Structure

```
repo/
├── README.md                   # This file
├── Makefile                    # Single-command entry points (12 targets)
├── requirements.txt            # Pinned Python dependencies
├── .env.example                # API key template (copy to .env)
├── .gitignore                  # Excludes .env, indices, caches
├── data/
│   ├── raw/                    # Source PDFs (17 papers)
│   ├── processed/              # FAISS index + BM25 corpus (auto-generated)
│   └── data_manifest.csv       # Source metadata (source_id, title, year, DOI, …)
├── src/
│   ├── ingest/
│   │   └── ingest.py           # PDF → section-aware chunks → FAISS + BM25
│   ├── rag/
│   │   └── rag.py              # Baseline vs. Enhanced RAG (4 enhancements)
│   ├── app/
│   │   └── app.py              # Streamlit research portal (5 pages)
│   └── eval/
│       └── run_eval.py         # 20-query evaluation runner
├── outputs/
│   ├── threads/                # Saved research threads (JSON)
│   └── artifacts/              # Generated evidence tables & synthesis memos
├── logs/
│   ├── eval_results.json       # Structured evaluation results
│   ├── run_logs.json           # Per-query retrieval + generation logs
│   ├── session_cache.json      # Full session cache for replay
│   ├── ingest_log.json         # Ingestion pipeline stats
│   ├── ingest.log              # Ingestion log (text)
│   └── rag.log                 # RAG pipeline log (text)
└── report/
    ├── phase1_report/          # Framing brief, prompt kit, evaluation sheet
    ├── phase2_report.pdf       # Evaluation report (3–5 pages)
    └── phase3_report/          # Final report (6–10 pages)
```

---

## System Robustness

| Concern | Implementation |
|---------|---------------|
| **API key security** | Loaded from `.env` via `python-dotenv`. Never hardcoded. `.gitignore` excludes `.env`. |
| **Idempotent ingestion** | Both Makefile and `ingest.py` check for existing FAISS + BM25 indices before running. Skips automatically unless `--force` is passed. Safe to call repeatedly with zero wasted API credits. |
| **API error handling** | All Anthropic and embedding calls wrapped in `tenacity` retry (3 attempts, exponential backoff 4–30s). |
| **Rate limiting** | Two-layer approach: (1) **Proactive** — configurable `API_CALL_DELAY` (1–2 s) sleeps between every LLM call in ingestion, querying, and evaluation loops to stay under rate ceilings. (2) **Reactive** — `tenacity` exponential backoff (4–30 s, 3 attempts) catches 429/5xx errors if the proactive delay is insufficient. |
| **Missing index** | Graceful error messages if FAISS or BM25 indices are missing (prompts user to run ingestion). |
| **Filter over-restriction** | Falls back to unfiltered results with a logged warning if metadata filters eliminate all candidates. |
| **Session caching** | Every pipeline run is persisted to `logs/session_cache.json` for offline replay. |

---

## Prompt Engineering Evolution

| Version | Phase | Key Changes |
|---------|-------|-------------|
| `v1.0` | Phase 1 | Baseline prompt — free-form answers, minimal citation guidance. |
| `v1.1-native-reranker` | Phase 2 early | Added structured citation format; single pipeline. |
| `v2.0-tabular-citations` | Phase 2 final | Enforced Markdown table output (`Claim \| Snippet \| Chunk ID`); A/B baseline vs. enhanced comparison. |
| `v3.0-hybrid-citations` | Phase 3 early | Added conflicting-evidence flagging; richer chunk headers (source, year, section); 3-enhancement pipeline. |
| `v3.1-hybrid-decompose-citations` | Phase 3 final | Added query decomposition for multi-hop synthesis; 4-enhancement pipeline; top-6 context for synthesis queries. |

---

## AI Usage Disclosure

| Tool | Usage | Manual Changes |
|------|-------|----------------|
| Claude (Anthropic) | Code scaffolding, prompt engineering iteration, report drafting assistance. | All code reviewed, tested, and modified. Evaluation scores assigned manually. |
| Claude (via RAG pipeline) | Header extraction during ingestion; answer generation during retrieval. | Outputs logged and evaluated per the rubric. |

---

## Authors

Farrukh Masood · Nishan Sah — Carnegie Mellon University, Policy Innovation Lab
