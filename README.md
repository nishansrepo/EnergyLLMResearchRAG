# Energy Sector Research Assistant (RAG Pipeline)

A Comparative Retrieval-Augmented Generation (RAG) system specialized for academic papers in the nexus of LLM and Energy domains. This tool ingests PDF documents, structures them by logical sections, and allows you to query them using a testing framework (Baseline RAG vs. Reranked RAG).

## Repository Structure

```
repo/
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── data/
│   ├── raw/                # PDF files go here
│   ├── processed/          # FAISS index (auto-generated)
│   └── data_manifest.csv   # Metadata for source documents
├── src/
│   ├── app/                # (Future) Phase 3 UI
│   ├── ingest/             # PDF parsing & vectorization scripts
        ├── ingest.py       # Ingestion code      
│   ├── rag/                # Retrieval & Generation logic
        ├── rag.py          # RAG code      
│   └── eval/               # (Future) Evaluation scripts
├── outputs/                # Artifacts and exports
├── logs/                   # A/B test run logs (JSON)
└── report/                 # Final reports
```

## Prerequisites

1. **Python 3.10+**
2. **API Keys**: You need a valid Anthropic API key (`ANTHROPIC_API_KEY`). Update the sections in ingest.py and rag.py

## Installation

1. Clone the repository (or create the folder structure).

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Data**:
   - PDF files are in `data/raw/`.
   - Update `data/data_manifest.csv` with the filenames and metadata.

## Usage

### 1. Ingest Documents

Run the ingestion script to parse PDFs and build the vector index.

```bash
# Run from repository root
python src/ingest/ingest.py
```

### 2. Execute a Query

Run the RAG pipeline to answer questions. You can pass your question as a command-line argument.

```bash
# Run from repository root
python src/rag/rag.py "What are the financial implications of deploying GAIA in Texas?"
```

## How it Works

### Ingestion (`src/ingest/`)
Uses Claude to identify logical sections in papers (headers), splits text by these sections, and indexes them using FAISS.

### RAG (`src/rag/`)

1. **Baseline**: Retrieves top 4 chunks via Cosine Similarity.
2. **Enhanced**: Retrieves top 15 chunks, re-ranks them using a Cross-Encoder, and picks the top 4.
3. **Generation**: Claude 3 Haiku generates answers with strict citations for both pipelines.
4. **Logging**: Results are saved to `logs/run_logs.json`.

## Features

- **A/B Testing Framework**: Compare Baseline RAG vs. Enhanced (Reranked) RAG performance
- **Section-Aware Chunking**: Intelligent document parsing that respects logical paper structure
- **Citation Tracking**: Strict citation requirements with chunk ID references
- **Academic Focus**: Optimized for energy sector research papers
- **Extensible Architecture**: Designed for future UI and evaluation modules

## Output

Query results are logged in JSON format to `logs/run_logs.json`, containing:
- Timestamp
- Query text
- Retrieved chunk IDs (for both baseline and enhanced pipelines)
- Generated responses with citations
- Prompt version used

## Future Enhancements

- **Phase 3 UI** (`src/app/`): Web interface for interactive querying
- **Better File Management**: The code is a little clunky at the moment
- **Evaluation Module** (`src/eval/`): Automated metrics for RAG performance assessment