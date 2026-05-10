import time
import hashlib
from collections import OrderedDict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    RAG_EMBEDDING_MODEL_NAME,
    RAG_CACHE_MAX_SIZE,
    RAG_DEFAULT_TOP_K,
    RAG_ENABLE_LOGGING,
)


DOCUMENTS = []
DOC_IDS = []
_SEEN_DOC_HASHES = set()

# FAISS index, embedding model, and caches live for the process lifetime.
index = None
_model = None
_cache = OrderedDict()
_query_embedding_cache = OrderedDict()


def _log(message: str):
    if RAG_ENABLE_LOGGING:
        print(f"[RAG] {message}")


def _load_model():
    # Lazy-load embeddings so import stays fast.
    global _model

    if _model is None:
        _log("Loading embedding model...")
        _model = SentenceTransformer(RAG_EMBEDDING_MODEL_NAME)
        _log("Model loaded.")

    return _model


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _make_cache_key(query: str, top_k: int) -> str:
    raw = f"{query.strip()}::{top_k}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _add_to_cache(key: str, value: dict):
    # Small LRU cache for repeated retrieval requests.
    _cache[key] = value
    _cache.move_to_end(key)

    if len(_cache) > RAG_CACHE_MAX_SIZE:
        _cache.popitem(last=False)


def _get_query_embedding(query: str):
    # Cache query embeddings separately from full retrieval results.
    if query in _query_embedding_cache:
        _query_embedding_cache.move_to_end(query)
        _log(f"Query embedding cache hit: {query}")
        return _query_embedding_cache[query]

    model = _load_model()
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    _query_embedding_cache[query] = query_embedding
    _query_embedding_cache.move_to_end(query)

    if len(_query_embedding_cache) > RAG_CACHE_MAX_SIZE:
        _query_embedding_cache.popitem(last=False)

    return query_embedding


def ingest_documents(docs):
    # Add only new, non-empty documents to the FAISS index.
    global DOCUMENTS, DOC_IDS, _SEEN_DOC_HASHES, index

    if not docs:
        _log("No documents provided for ingestion.")
        return {
            "success": False,
            "added": 0,
            "total_documents": len(DOCUMENTS),
            "message": "No documents provided.",
        }

    model = _load_model()

    new_docs = []
    new_doc_ids = []

    for doc in docs:
        if not doc or not doc.strip():
            continue

        doc = doc.strip()
        doc_hash = _hash_text(doc)

        if doc_hash in _SEEN_DOC_HASHES:
            continue

        doc_id = f"doc_{len(DOCUMENTS) + len(new_docs)}"

        new_docs.append(doc)
        new_doc_ids.append(doc_id)
        _SEEN_DOC_HASHES.add(doc_hash)

    if not new_docs:
        _log("No new documents added. All were empty or duplicates.")
        return {
            "success": True,
            "added": 0,
            "total_documents": len(DOCUMENTS),
            "message": "No new documents added.",
        }

    embeddings = model.encode(new_docs)
    embeddings = np.array(embeddings).astype("float32")

    if index is None:
        index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)

    DOCUMENTS.extend(new_docs)
    DOC_IDS.extend(new_doc_ids)

    _log(f"Ingested {len(new_docs)} new documents. Total: {len(DOCUMENTS)}")

    return {
        "success": True,
        "added": len(new_docs),
        "total_documents": len(DOCUMENTS),
        "doc_ids": new_doc_ids,
    }


def retrieve(query: str, top_k: int = None) -> dict:
    # Return the most relevant documents and combined context for the LLM.
    global index, DOCUMENTS, DOC_IDS, _cache

    start_time = time.time()

    if top_k is None:
        top_k = RAG_DEFAULT_TOP_K

    if not query or not query.strip():
        return {
            "success": False,
            "query": query,
            "context": "",
            "results": [],
            "top_k": top_k,
            "cached": False,
            "latency": 0,
            "error": "empty_query",
        }

    if index is None or len(DOCUMENTS) == 0:
        _log("No documents available.")
        return {
            "success": False,
            "query": query,
            "context": "",
            "results": [],
            "top_k": top_k,
            "cached": False,
            "latency": time.time() - start_time,
            "error": "no_documents",
        }

    query = query.strip()
    top_k = min(top_k, len(DOCUMENTS))

    cache_key = _make_cache_key(query, top_k)

    if cache_key in _cache:
        cached_result = _cache[cache_key]
        _cache.move_to_end(cache_key)

        cached_copy = cached_result.copy()
        cached_copy["cached"] = True
        cached_copy["latency"] = time.time() - start_time

        _log(f"Cache hit for query: {query}")
        return cached_copy

    query_embedding = _get_query_embedding(query)

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for rank, doc_index in enumerate(indices[0]):
        if doc_index < 0 or doc_index >= len(DOCUMENTS):
            continue

        distance = float(distances[0][rank])

        results.append(
            {
                "rank": rank + 1,
                "doc_id": DOC_IDS[doc_index],
                "text": DOCUMENTS[doc_index],
                "distance": distance,
                "score": 1 / (1 + distance),
            }
        )

    context = "\n".join(item["text"] for item in results)

    response = {
        "success": True,
        "query": query,
        "context": context,
        "results": results,
        "top_k": top_k,
        "cached": False,
        "latency": time.time() - start_time,
        "error": None,
    }

    _add_to_cache(cache_key, response)

    _log(
        f"Retrieved top-{top_k} documents | "
        f"latency={response['latency']:.4f}s"
    )

    return response


def retrieve_context(query: str, top_k: int = None) -> str:
    """
    Backward-compatible function for existing worker code.
    Returns only the combined context string.
    """
    result = retrieve(query, top_k)
    return result["context"]
