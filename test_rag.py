from workers.rag import ingest_documents, retrieve_context

# Step 1: Add documents (this is ingestion)
ingest_documents([
    "Load balancing distributes requests across workers.",
    "Round Robin assigns tasks in circular order.",
    "RAG retrieves relevant context before LLM inference.",
    "Fault tolerance ensures system reliability.",
    "GPU workers execute AI tasks in parallel."
])

# Step 2: Query
query = "How does load balancing work?"

# Step 3: Retrieve context
context = retrieve_context(query)

print("\nRetrieved Context:\n")
print(context)