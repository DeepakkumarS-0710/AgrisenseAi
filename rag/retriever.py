import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGRetriever:
    def __init__(self, index_dir="rag/index"):
        self.index_dir = index_dir
        self.index = None
        self.metadata = []
        self.model = None
        self._load_resources()

    def _load_resources(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        index_path = os.path.join(self.index_dir, "faiss.index")
        metadata_path = os.path.join(self.index_dir, "metadata.json")

        if not os.path.exists(index_path):
            alt_dir = os.path.join(base_dir, self.index_dir)
            index_path = os.path.join(alt_dir, "faiss.index")
            metadata_path = os.path.join(alt_dir, "metadata.json")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print(f"Warning: RAG index or metadata missing at {index_path}. Run rag/ingest.py first.")
            return

        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def retrieve(self, query, top_k=3):
        if self.index is None or not self.metadata or self.model is None:
            # Fallback mock if index not yet generated
            return [{
                "source_doc": "agrisense_kb.txt",
                "text": f"General guidance for query: {query}",
                "score": 0.0
            }]

        query_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            score = float(scores[0][i])
            if idx >= 0 and idx < len(self.metadata):
                item = self.metadata[idx]
                results.append({
                    "source_doc": item.get("source_doc", "Unknown Doc"),
                    "text": item.get("text", ""),
                    "score": round(score, 4)
                })

        return results

_retriever_instance = None

def retrieve_knowledge(query, top_k=3):
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever()
    return _retriever_instance.retrieve(query, top_k=top_k)

if __name__ == "__main__":
    print("Testing RAG Retriever...")
    res = retrieve_knowledge("Apple Scab symptoms and treatment")
    print("Retrieved context:", res)
