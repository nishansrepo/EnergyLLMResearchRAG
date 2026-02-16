# src/rag/rag.py
import os
import json
import sys
from datetime import datetime
from pathlib import Path

# ==========================================
# HARDCODED API KEY
# Replace with your actual Anthropic key before running.
# ==========================================
os.environ["ANTHROPIC_API_KEY"] = ""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# ==========================================
# PATH SETUP
# ==========================================
# Resolves paths relative to this script so it runs from anywhere
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
FAISS_INDEX_PATH = DATA_DIR / "processed" / "faiss_index"

PROMPT_VERSION = "v2.0-tabular-citations"
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert AI research assistant for the energy sector. 
Your task is to answer the user's query using ONLY the provided context chunks.

TRUST BEHAVIOR & CONSTRAINTS:
1. If the context does not contain the answer, explicitly state exactly: "I cannot find evidence for this claim in the provided context." Do NOT hallucinate.
2. Every claim or fact you output MUST be cited.
3. Your citation MUST contain both the exact Chunk ID and a short, direct text snippet from the context that supports the claim.
4. Format your final answer as a Markdown table with the following columns:
   | Claim / Information | Supporting Text Snippet | Chunk ID |

Context:
{context}"""),
    ("human", "{question}")
])

def format_docs(docs):
    formatted_chunks = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "UNKNOWN_ID")
        formatted_chunks.append(f"--- Chunk ID: [{chunk_id}] ---\n{doc.page_content}")
    return "\n\n".join(formatted_chunks)

def log_interaction(query, baseline_docs, baseline_response, enhanced_docs, enhanced_response):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "run_logs.json"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "prompt_version": PROMPT_VERSION,
        "baseline_rag": {
            "retrieved_chunk_ids": [doc.metadata.get("chunk_id", "UNKNOWN_ID") for doc in baseline_docs],
            "llm_output": baseline_response
        },
        "enhanced_rag": {
            "retrieved_chunk_ids": [doc.metadata.get("chunk_id", "UNKNOWN_ID") for doc in enhanced_docs],
            "llm_output": enhanced_response
        }
    }
    
    logs = []
    if log_file.exists():
        with open(log_file, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
                
    logs.append(log_entry)
    
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)
        
    print(f"\n[+] Interaction logged successfully to {log_file}")

def run_comparative_rag(query: str):
    print(f"\nProcessing Query: '{query}'\n")
    
    print("Loading FAISS index...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(
            str(FAISS_INDEX_PATH), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except RuntimeError as e:
        print(f"\n[!] Error loading Vector Store from {FAISS_INDEX_PATH}.")
        print("Did you run 'src/ingest/ingest.py' first?")
        print(f"Details: {e}")
        return
    
    # --- Baseline ---
    print("\n--- Executing Baseline Naive RAG ---")
    baseline_docs = vectorstore.similarity_search(query, k=4)
    print(f"Retrieved top {len(baseline_docs)} chunks using pure semantic similarity.")
    
    # --- Enhanced ---
    print("\n--- Executing Enhanced RAG (Cross-Encoder Reranking) ---")
    initial_docs = vectorstore.similarity_search(query, k=15)
    
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    query_doc_pairs = [[query, doc.page_content] for doc in initial_docs]
    scores = cross_encoder.predict(query_doc_pairs)
    
    scored_docs = zip(scores, initial_docs)
    sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
    enhanced_docs = [doc for score, doc in sorted_docs[:4]]
    print(f"Reranked 15 chunks down to top {len(enhanced_docs)} using Cross-Encoder.")
    
    # --- Generation ---
    print("\nGenerating responses via Claude...")
    try:
        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
        rag_chain = RAG_PROMPT | llm | StrOutputParser()
        
        print("  -> Generating Baseline Response...")
        baseline_response = rag_chain.invoke({
            "context": format_docs(baseline_docs),
            "question": query
        })
        
        print("  -> Generating Enhanced Response...")
        enhanced_response = rag_chain.invoke({
            "context": format_docs(enhanced_docs),
            "question": query
        })
        
        log_interaction(query, baseline_docs, baseline_response, enhanced_docs, enhanced_response)
        
        print("\n" + "="*60)
        print("BASELINE RAG OUTPUT")
        print("="*60)
        print(baseline_response)
        
        print("\n" + "="*60)
        print("ENHANCED RAG OUTPUT")
        print("="*60)
        print(enhanced_response)
        
    except Exception as e:
        print(f"\n[!] Error during generation: {e}")

if __name__ == "__main__":
    default_query = "According to the provided texts, what is the exact financial cost in USD of deploying GAIA across the entire Texas ERCOT power grid?"
    
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        print(f"No query provided. Using default sample query.")
        user_query = default_query
        
    run_comparative_rag(user_query)