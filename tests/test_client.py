# Client workload and end-to-end scheduler smoke tests.
from scheduler.scheduler import Scheduler
from load_balancer.load_balancer import LoadBalancer
from workers.gpu_worker import GPUWorker
from client.client import run_client
from client.client import generate_query_workload
from common.text_utils import keyword_signature
from workers.rag import ingest_documents


def test_generate_query_workload_size_and_variety():
    queries = generate_query_workload(size=1000, seed=123)

    assert len(queries) == 1000
    assert len(set(queries)) >= 700
    assert len({keyword_signature(query) for query in queries}) >= 650


def test_generate_query_workload_repeat_ratio():
    queries = generate_query_workload(
        size=100,
        seed=123,
        unique_ratio=0.2,
        repeat_pool_size=5,
    )

    assert len(queries) == 100
    assert 5 <= len(set(queries)) <= 25

def run_test(strategy, num_users):
    print(f"\n=== Testing {strategy} with {num_users} users ===")

    ingest_documents([
        "Load balancing distributes incoming requests across multiple servers.",
        "Round Robin assigns tasks in circular order.",
        "Least Connections sends requests to the least loaded worker.",
        "Fault tolerance allows systems to recover from failures.",
        "RAG retrieves relevant documents before generating answers.",
        "LLM generates answers using query and retrieved context."
    ])

    # Create workers
    workers = [GPUWorker(i) for i in range(3)]

    # Create load balancer
    lb = LoadBalancer(workers, strategy=strategy)

    # Create scheduler
    scheduler = Scheduler(lb)

    # Run test
    run_client(scheduler, num_users)


if __name__ == "__main__":
    for strategy in ["round_robin", "least_connections", "load_aware"]:
        for users in [50, 100, 200]:
            run_test(strategy, users)
