# ============================================================================
# Makefile — Energy Sector Research Assistant (RAG Pipeline)
# Single-command entry points for installation, ingestion, querying, and demo.
# ============================================================================

.PHONY: install ingest query eval app demo replay clean help

PYTHON ?= python3
PIP    ?= pip3

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:  ## Install all Python dependencies
	$(PIP) install -r requirements.txt

env-check:  ## Verify .env file exists
	@if [ ! -f .env ]; then \
		echo "ERROR: .env file not found."; \
		echo "Copy .env.example to .env and add your ANTHROPIC_API_KEY."; \
		echo "  cp .env.example .env"; \
		exit 1; \
	fi

# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

ingest: env-check  ## Ingest PDFs (skips if FAISS + BM25 indices already exist)
	@if [ -d data/processed/faiss_index ] && [ -f data/processed/bm25_corpus.pkl ]; then \
		echo "=== Indices already exist. Skipping ingestion. (use 'make ingest-force' to rebuild) ==="; \
	else \
		echo "=== Running Ingestion Pipeline ==="; \
		$(PYTHON) src/ingest/ingest.py; \
	fi

ingest-force: env-check  ## Force re-ingestion (rebuilds indices from scratch)
	@echo "=== Force Re-Ingestion ==="
	$(PYTHON) src/ingest/ingest.py --force

query: env-check  ## Run a single query (usage: make query Q="your question here")
	@echo "=== Running RAG Query ==="
	$(PYTHON) src/rag/rag.py "$(Q)"

query-filtered: env-check  ## Filtered query (usage: make query-filtered Q="..." YEAR=2024 TYPE=journal)
	$(PYTHON) src/rag/rag.py "$(Q)" --year $(YEAR) --source-type $(TYPE)

eval: env-check  ## Run the full 20-query evaluation set
	@echo "=== Running Evaluation Suite ==="
	$(PYTHON) src/eval/run_eval.py

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app:  ## Launch the Streamlit research portal (auto-detects API key)
	@echo "=== Launching Research Portal ==="
	@echo "  Open http://localhost:8501 in your browser"
	streamlit run src/app/app.py --server.port 8501

# ---------------------------------------------------------------------------
# Demo & Replay (no API key required)
# ---------------------------------------------------------------------------

replay:  ## Replay cached demo sessions (no API key needed)
	@echo "=== Replaying Cached Sessions ==="
	$(PYTHON) src/rag/rag.py --replay "$(Q)"

demo-replay:  ## Launch the app in demo mode (cached results, no API key)
	@echo "=== Launching Portal in Demo Mode ==="
	@echo "  Open http://localhost:8501 in your browser"
	DEMO_MODE=true streamlit run src/app/app.py --server.port 8501

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

clean:  ## Remove generated indices and logs (keeps raw data)
	rm -rf data/processed/faiss_index
	rm -f data/processed/bm25_corpus.pkl
	rm -f logs/run_logs.json logs/session_cache.json logs/ingest.log logs/rag.log
	@echo "Cleaned generated files."

all: install ingest eval  ## Full pipeline: install → ingest → evaluate

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
