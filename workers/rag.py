import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Global storage
DOCUMENTS = []
index = None
_model = None

# Load embedding model (once)
def _load_model():
    global _model
    if _model is None:
        print("[RAG] Loading embedding model...")
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("[RAG] Model loaded.")
    return _model


# Ingest documents into FAISS
def ingest_documents(docs):
    global DOCUMENTS, index

    if not docs:
        return

    model = _load_model()

    # Convert documents to embeddings
    embeddings = model.encode(docs)
    embeddings = np.array(embeddings).astype("float32")

    # Create index if not exists
    if index is None:
        index = faiss.IndexFlatL2(embeddings.shape[1])

    # Add embeddings to index
    index.add(embeddings)

    # Store original documents
    DOCUMENTS.extend(docs)

    print(f"[RAG] Ingested {len(docs)} documents. Total: {len(DOCUMENTS)}")


# Retrieve context using FAISS
def retrieve_context(query: str, top_k: int = 3) -> str:
    global index, DOCUMENTS

    if not query or not query.strip():
        return ""

    if index is None or len(DOCUMENTS) == 0:
        print("[RAG] No documents available.")
        return ""

    model = _load_model()

    # Encode query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search in FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve documents
    results = [DOCUMENTS[i] for i in indices[0] if i < len(DOCUMENTS)]

    print(f"[RAG] Retrieved top-{top_k} documents for query: {query}")

    return "\n".join(results)