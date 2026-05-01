from common.types import Request
from concurrent.futures import ThreadPoolExecutor
import time
import config
import random
import statistics

# Query Pool (simulate real users)
QUERIES = [
    "What is AI?",
    "Explain load balancing",
    "What is RAG?",
    "How does fault tolerance work?",
    "What is distributed computing?",
]

# Wrapper (abstraction layer)
def send_to_scheduler(scheduler, request):
    return scheduler.handle_request(request)


# Send single request
def send_request(scheduler, request_id):
    query = random.choice(QUERIES)
    request = Request(id=request_id, query=query)

    start = time.time()

    try:
        response = send_to_scheduler(scheduler, request)

        success = response.result != "FAILED"
        latency = response.latency

    except Exception:
        success = False
        latency = time.time() - start

    return {
        "success": success,
        "latency": latency
    }


# Run load test
def run_client(scheduler, num_requests=None):
    if num_requests is None:
        num_requests = config.NUM_USERS

    results = []

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = []

        for i in range(num_requests):
            futures.append(executor.submit(send_request, scheduler, i))
            time.sleep(config.REQUEST_DELAY)

        # wait for all requests
        for future in futures:
            results.append(future.result())

    end_time = time.time()
    total_time = end_time - start_time

    # Metrics Calculation
    total_requests = len(results)
    successes = sum(1 for r in results if r["success"])
    failures = total_requests - successes

    latencies = [r["latency"] for r in results if r["latency"] is not None]

    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    # P95 calculation
    if len(latencies) >= 100:
        p95_latency = statistics.quantiles(latencies, n=100)[94]
    else:
        p95_latency = max(latencies, default=0)

    throughput = total_requests / total_time if total_time > 0 else 0

    # Output Report
    print("\n===== PERFORMANCE REPORT =====")
    print(f"Total requests: {total_requests}")
    print(f"Successful: {successes}")
    print(f"Failed: {failures}")
    print(f"Average latency: {avg_latency:.2f}s")
    print(f"P95 latency: {p95_latency:.2f}s")
    print(f"Throughput: {throughput:.2f} req/sec")
    print("================================\n")