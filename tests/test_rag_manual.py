from workers.rag import ingest_documents, retrieve, retrieve_context


def test_ingest_documents():
    result = ingest_documents([
        "Load balancing distributes incoming requests across multiple servers.",
        "Fault tolerance allows a system to continue working after worker failure.",
        "RAG retrieves relevant documents before generating an answer.",
        "GPU utilization measures how much GPU processing capacity is being used.",
        "Load balancing distributes incoming requests across multiple servers."
    ])

    print("\nIngest result:", result)

    assert result["success"] is True
    assert result["added"] == 4
    assert result["total_documents"] >= 4
    assert "doc_ids" in result


def test_retrieve_with_metadata():
    ingest_documents([
        "Load balancing distributes incoming requests across multiple servers.",
        "Fault tolerance allows a system to continue working after worker failure.",
        "RAG retrieves relevant documents before generating an answer.",
    ])

    result = retrieve("What is load balancing?", top_k=2)

    print("\nRetrieve result:", result)

    assert result["success"] is True
    assert result["context"] != ""
    assert len(result["results"]) == 2
    assert "latency" in result
    assert "score" in result["results"][0]
    assert "distance" in result["results"][0]
    assert "doc_id" in result["results"][0]
    assert result["cached"] is False


def test_retrieve_context_backward_compatible():
    ingest_documents([
        "RAG means Retrieval-Augmented Generation.",
        "It improves LLM answers using external knowledge."
    ])

    context = retrieve_context("What is RAG?", top_k=1)

    print("\nContext:", context)

    assert isinstance(context, str)
    assert len(context) > 0


def test_cache():
    ingest_documents([
        "GPU utilization measures how much GPU capacity is currently used."
    ])

    first = retrieve("What is GPU utilization?", top_k=1)
    second = retrieve("What is GPU utilization?", top_k=1)

    print("\nFirst retrieval:", first)
    print("\nSecond retrieval:", second)

    assert first["success"] is True
    assert second["success"] is True
    assert second["cached"] is True


def test_empty_query():
    result = retrieve("", top_k=1)

    print("\nEmpty query result:", result)

    assert result["success"] is False
    assert result["error"] == "empty_query"
    assert result["context"] == ""