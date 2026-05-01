from scheduler.scheduler import Scheduler
from load_balancer.load_balancer import LoadBalancer
from workers.gpu_worker import GPUWorker
from client.client import run_client
from workers.rag import ingest_documents

ingest_documents([
    "Load balancing distributes incoming requests across multiple servers.",
    "Round Robin assigns tasks in circular order.",
    "Least Connections sends requests to the least loaded worker.",
    "Fault tolerance allows systems to recover from failures.",
    "RAG retrieves relevant documents before generating answers.",
    "LLM generates answers using query and retrieved context."
])

def run_test(strategy, num_users):
    print(f"\n=== Testing {strategy} with {num_users} users ===")

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