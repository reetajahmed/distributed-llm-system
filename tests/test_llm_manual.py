from llm.model import run_llm, run_llm_with_metrics, run_llm_batch


def test_basic_llm():
    answer = run_llm(
        "What is distributed computing?",
        request_id="basic_req_1"
    )

    print("\nBasic answer:", answer)

    assert isinstance(answer, str)
    assert len(answer) > 0


def test_llm_with_metrics():
    result = run_llm_with_metrics(
        query="What is load balancing?",
        context="Load balancing distributes incoming requests across multiple servers.",
        request_id="metrics_req_1"
    )

    print("\nMetrics result:", result)

    assert result["request_id"] == "metrics_req_1"
    assert result["success"] is True
    assert isinstance(result["answer"], str)
    assert result["latency"] >= 0
    assert result["input_tokens"] > 0
    assert result["output_tokens"] > 0
    assert result["device"] in ["cpu", "cuda"]
    assert result["error"] is None


def test_cache():
    query = "What is GPU utilization?"
    context = "GPU utilization measures how much of the GPU processing capacity is used."

    first = run_llm_with_metrics(
        query=query,
        context=context,
        request_id="cache_req_1"
    )

    second = run_llm_with_metrics(
        query=query,
        context=context,
        request_id="cache_req_2"
    )

    print("\nFirst call:", first)
    print("\nSecond call:", second)

    assert first["request_id"] == "cache_req_1"
    assert second["request_id"] == "cache_req_2"
    assert first["success"] is True
    assert second["success"] is True
    assert second["cached"] is True


def test_batch():
    requests = [
        {
            "request_id": "batch_req_1",
            "query": "What is RAG?",
            "context": "RAG means Retrieval-Augmented Generation."
        },
        {
            "request_id": "batch_req_2",
            "query": "What is fault tolerance?",
            "context": "Fault tolerance allows a system to continue operating after failures."
        }
    ]

    results = run_llm_batch(requests)

    print("\nBatch results:", results)

    assert len(results) == 2
    assert results[0]["request_id"] == "batch_req_1"
    assert results[1]["request_id"] == "batch_req_2"
    assert all(result["success"] is True for result in results)
    assert all(isinstance(result["answer"], str) for result in results)