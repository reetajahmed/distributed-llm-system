# Provides retrieve_context(query) for RAG pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


DOCUMENTS = [
    "Load balancing distributes incoming requests across multiple workers to improve performance and prevent overload.",
    "Round Robin assigns tasks to workers one by one in circular order.",
    "Least Connections sends new requests to the worker with the fewest active tasks.",
    "Load-aware routing selects workers based on current load, availability, and performance metrics.",

    "The scheduler receives requests and assigns them to available worker nodes.",
    "The scheduler tracks task status, worker availability, and request completion.",
    "The scheduler should reassign failed tasks to active workers to avoid request loss.",

    "A GPU worker node executes LLM inference tasks and returns the result to the scheduler.",
    "GPU workers process requests in parallel to improve throughput.",
    "Each worker should expose metrics such as active task count, status, and average latency.",

    "RAG stands for Retrieval-Augmented Generation.",
    "RAG retrieves relevant context from a knowledge base before sending the query to the LLM.",
    "RAG improves answer accuracy by grounding the LLM response in retrieved information.",
    "The RAG module returns context, not the final answer.",

    "LLM inference means generating an answer using a trained language model.",
    "The LLM receives the user query and retrieved context, then generates a final response.",
    "The LLM model should be loaded once and reused across requests to reduce latency.",
    "Reloading the LLM model for every request causes high latency and memory usage.",

    "Fault tolerance means detecting failed workers and reassigning their tasks to active workers.",
    "Worker failure should not cause request loss.",
    "Task reassignment maintains processing continuity during partial system failure.",
    "The load balancer should continue routing requests even if some workers fail.",

    "The client layer simulates many concurrent users sending AI requests to the system.",
    "The system should be tested with increasing loads such as 100, 500, and 1000 concurrent users.",
    "Performance metrics include latency, throughput, worker utilization, and failure recovery time.",
]


_model = None
_doc_embeddings = None


def _load_rag_model():
    global _model, _doc_embeddings

    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _doc_embeddings = _model.encode(DOCUMENTS)

    return _model, _doc_embeddings


def retrieve_context(query: str, top_k: int = 3) -> str:
    if not query or not query.strip():
        return ""

    model, doc_embeddings = _load_rag_model()

    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    top_indices = np.argsort(scores)[-top_k:][::-1]
    selected_docs = [DOCUMENTS[i] for i in top_indices]

    return "\n".join(selected_docs)