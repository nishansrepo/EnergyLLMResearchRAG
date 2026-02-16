# src/ingest/ingest_pipeline.py
import pandas as pd
import os
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic 

# ==========================================
# HARDCODED API KEY
# Replace the string below with your actual Anthropic key.
# ==========================================
os.environ["ANTHROPIC_API_KEY"] = ""

# 1. Define the Expected Output Structure for the LLM
class PaperOutline(BaseModel):
    headers: list[str] = Field(
        description="Chronological list of the exact section and subsection titles found in the paper text."
    )

def extract_sections_with_llm(raw_text: str, llm: ChatAnthropic) -> list[str]:
    """Uses an LLM to read the paper and extract its structural headers."""
    # LangChain automatically maps the Pydantic model to Claude's tool-use format
    structured_llm = llm.with_structured_output(PaperOutline)
    
    prompt = f"""
    You are an expert academic parser. Review the following text from an academic paper. 
    Extract a chronological list of all major Section and Subsection titles exactly as they 
    appear in the text (e.g., "1. Introduction", "II. Proposed Methodology", "3.1 Data Collection").
    
    Paper Text (Truncated):
    {raw_text[:60000]} 
    """
    
    try:
        result = structured_llm.invoke(prompt)
        return result.headers
    except Exception as e:
        print(f"LLM Extraction failed: {e}")
        return []

def split_text_by_headers(text: str, headers: list[str]) -> list[dict]:
    """Splits the raw text into sections based on the exact string matches of the headers."""
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

def run_llm_section_aware_ingestion():
    df = pd.read_csv('data/data_manifest.csv')
    df.fillna("", inplace=True)
    
    # 2. Configure Tools
    # Using Claude 3 Haiku for cheap, fast, long-context extraction
    llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    all_chunks = []
    
    for _, row in df.iterrows():
        pdf_path = os.path.join("data/raw", row['raw_path'])
        if not os.path.exists(pdf_path):
            print(f"Skipping {row['source_id']}: File not found.")
            continue
            
        print(f"\nProcessing {row['source_id']}...")
        
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()
        raw_full_text = "\n".join([page.page_content for page in pages])
        
        print("  - Extracting logical headers via Claude...")
        extracted_headers = extract_sections_with_llm(raw_full_text, llm)
        print(f"  - Found {len(extracted_headers)} headers.")
        
        logical_sections = split_text_by_headers(raw_full_text, extracted_headers)
        
        document_chunks = []
        
        for section in logical_sections:
            sub_chunks = text_splitter.create_documents([section["content"]])
            
            for chunk in sub_chunks:
                chunk.metadata.update({
                    "source_id": row['source_id'],
                    "title": row['title'],
                    "year": row['year'],
                    "tags": row['tags'],
                    "Section": section["header"]
                })
                document_chunks.append(chunk)

        for i, chunk in enumerate(document_chunks):
            chunk.metadata["chunk_id"] = f"{row['source_id']}_chunk_{i}"
            
        all_chunks.extend(document_chunks)
        print(f"  - Generated {len(document_chunks)} chunks.")
            
    print(f"\nTotal section-aware chunks created across corpus: {len(all_chunks)}")
    
    print("\nBuilding FAISS index (this may take a minute)...")
    os.makedirs("data/processed", exist_ok=True)
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local("data/processed/faiss_index")
    print("Vector database saved successfully!")

if __name__ == "__main__":
    run_llm_section_aware_ingestion()