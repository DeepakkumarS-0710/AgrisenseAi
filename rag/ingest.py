import os
import glob
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pypdf

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif ext == ".pdf":
        try:
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    text = text.strip()
    if not text:
        return chunks
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def build_rag_index(docs_dir="rag/documents", output_dir="rag/index"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_docs_dir = os.path.join(base_dir, docs_dir) if not os.path.isabs(docs_dir) else docs_dir
    abs_output_dir = os.path.join(base_dir, output_dir) if not os.path.isabs(output_dir) else output_dir
    
    os.makedirs(abs_output_dir, exist_ok=True)
    
    print(f"Reading documents from: {abs_docs_dir}")
    doc_files = glob.glob(os.path.join(abs_docs_dir, "*.*"))
    supported_files = [f for f in doc_files if f.lower().endswith(('.txt', '.pdf'))]
    
    if not supported_files:
        print(f"No .txt or .pdf files found in {abs_docs_dir}")
        return
        
    all_chunks = []
    metadata = []
    
    for file_path in supported_files:
        doc_name = os.path.basename(file_path)
        print(f"Processing document: {doc_name}")
        raw_text = extract_text_from_file(file_path)
        chunks = chunk_text(raw_text, chunk_size=400, overlap=80)
        
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({
                "chunk_id": len(all_chunks) - 1,
                "source_doc": doc_name,
                "text": chunk
            })
            
    print(f"\nExtracted total {len(all_chunks)} text chunks across {len(supported_files)} documents.")
    
    print("Generating sentence embeddings using 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    
    # Normalize for cosine similarity / L2 distance
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Inner product with normalized vectors = Cosine Similarity
    index.add(embeddings)
    
    # Save index & metadata
    index_path = os.path.join(abs_output_dir, "faiss.index")
    metadata_path = os.path.join(abs_output_dir, "metadata.json")
    
    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nFAISS index successfully saved to: {index_path}")
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    build_rag_index()
