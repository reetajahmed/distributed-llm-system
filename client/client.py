from common.types import Request
from common.text_utils import keyword_signature
from concurrent.futures import ThreadPoolExecutor
import time
import config
import random
import statistics

QUERY_TEMPLATES = {
    "ai": [
        "What is AI?",
        "Explain artificial intelligence.",
        "Define AI in simple terms.",
        "How does artificial intelligence work?",
        "What can AI systems do?",
        "Tell me about artificial intelligence.",
    ],
    "load_balancing": [
        "Explain load balancing.",
        "What is load balancing?",
        "How does a load balancer distribute requests?",
        "Why do distributed systems use load balancing?",
        "Describe least connections and round robin routing.",
        "How can traffic be shared across workers?",
    ],
    "rag": [
        "What is RAG?",
        "Explain retrieval augmented generation.",
        "How does RAG retrieve context?",
        "Why does RAG help LLM answers?",
        "Describe retrieval before generation.",
        "What is retrieval augmented generation used for?",
    ],
    "fault_tolerance": [
        "How does fault tolerance work?",
        "What is fault tolerance?",
        "Explain failure recovery in distributed systems.",
        "How can a system continue after worker failure?",
        "Describe retry and reassignment after failure.",
        "Why is fault tolerance important?",
    ],
    "distributed_computing": [
        "What is distributed computing?",
        "Explain distributed computing.",
        "How do multiple nodes split work?",
        "Why use distributed systems?",
        "Describe computation across several machines.",
        "What makes a system distributed?",
    ],
}

QUERY_SCENARIOS = {
    "ai": {
        "subjects": [
            "model inference",
            "text classification",
            "chatbot answers",
            "decision support",
            "automation",
            "recommendation systems",
            "fraud detection",
            "document summarization",
            "customer support",
            "data analysis",
        ],
        "contexts": [
            "healthcare",
            "finance",
            "education",
            "retail",
            "manufacturing",
            "software engineering",
            "logistics",
            "security monitoring",
            "research teams",
            "small businesses",
        ],
        "angles": [
            "benefits",
            "risks",
            "latency impact",
            "accuracy tradeoffs",
            "cost control",
            "failure modes",
            "monitoring needs",
            "privacy concerns",
            "scaling limits",
            "quality checks",
        ],
    },
    "load_balancing": {
        "subjects": [
            "round robin",
            "least connections",
            "load-aware routing",
            "weighted routing",
            "queue depth",
            "active connections",
            "GPU capacity",
            "worker latency",
            "request bursts",
            "traffic spikes",
        ],
        "contexts": [
            "three worker nodes",
            "mixed GPU capacities",
            "high concurrency",
            "slow model inference",
            "cache-heavy traffic",
            "cold-cache traffic",
            "regional workers",
            "batch requests",
            "interactive users",
            "background jobs",
        ],
        "angles": [
            "worker selection",
            "fairness",
            "throughput",
            "tail latency",
            "queue buildup",
            "overload prevention",
            "capacity planning",
            "health checks",
            "routing metrics",
            "failure recovery",
        ],
    },
    "rag": {
        "subjects": [
            "document retrieval",
            "embedding search",
            "FAISS indexing",
            "top-k context",
            "query embeddings",
            "context quality",
            "duplicate documents",
            "cache hits",
            "semantic search",
            "answer grounding",
        ],
        "contexts": [
            "technical documentation",
            "support tickets",
            "company policies",
            "research notes",
            "incident reports",
            "product manuals",
            "engineering runbooks",
            "meeting transcripts",
            "knowledge bases",
            "user guides",
        ],
        "angles": [
            "retrieval accuracy",
            "latency",
            "index updates",
            "stale context",
            "chunk quality",
            "ranking",
            "deduplication",
            "cache invalidation",
            "answer relevance",
            "failure handling",
        ],
    },
    "fault_tolerance": {
        "subjects": [
            "worker crash",
            "timeout",
            "retry",
            "request reassignment",
            "health check failure",
            "partial outage",
            "slow worker",
            "network error",
            "failed inference",
            "queue overflow",
        ],
        "contexts": [
            "distributed LLM workers",
            "remote HTTP workers",
            "GPU service",
            "scheduler process",
            "load balancer",
            "high request volume",
            "burst traffic",
            "multi-node deployment",
            "cache-backed responses",
            "long-running inference",
        ],
        "angles": [
            "recovery",
            "retries",
            "user impact",
            "availability",
            "latency penalty",
            "failure detection",
            "safe fallback",
            "metrics",
            "capacity loss",
            "permanent failure",
        ],
    },
    "distributed_computing": {
        "subjects": [
            "multiple processes",
            "remote workers",
            "HTTP services",
            "scheduler coordination",
            "parallel execution",
            "shared cache",
            "independent worker state",
            "horizontal scaling",
            "network calls",
            "service boundaries",
        ],
        "contexts": [
            "one machine",
            "multiple servers",
            "Docker containers",
            "cloud VMs",
            "GPU nodes",
            "local development",
            "production deployment",
            "mixed hardware",
            "separate regions",
            "service mesh",
        ],
        "angles": [
            "scalability",
            "fault isolation",
            "throughput",
            "latency overhead",
            "coordination",
            "deployment",
            "observability",
            "resource usage",
            "data consistency",
            "bottlenecks",
        ],
    },
}

REALISTIC_QUERY_PATTERNS = [
    "How does {subject} affect {angle} in {context}?",
    "Explain {subject} for {context} when optimizing {angle}.",
    "What should we monitor for {subject} in {context}?",
    "Why is {subject} important for {angle} in {context}?",
    "Compare {subject} and {angle} for {context}.",
    "What can go wrong with {subject} during {context}?",
    "How would you improve {angle} when using {subject} for {context}?",
    "Give a practical example of {subject} in {context}.",
    "What is the tradeoff between {subject} and {angle} in {context}?",
    "When should a system use {subject} for {context}?",
]


def _generate_unique_queries(rng, count):
    candidates = []

    for topic, parts in QUERY_SCENARIOS.items():
        for subject in parts["subjects"]:
            for context in parts["contexts"]:
                for angle in parts["angles"]:
                    pattern = rng.choice(REALISTIC_QUERY_PATTERNS)
                    candidates.append(
                        pattern.format(
                            subject=subject,
                            context=context,
                            angle=angle,
                        )
                    )

    rng.shuffle(candidates)
    return candidates[:count]


def generate_query_workload(size=None, seed=None, unique_ratio=None, repeat_pool_size=None):
    if size is None:
        size = config.QUERY_WORKLOAD_SIZE
    if seed is None:
        seed = config.QUERY_RANDOM_SEED
    if unique_ratio is None:
        unique_ratio = config.QUERY_UNIQUE_RATIO
    if repeat_pool_size is None:
        repeat_pool_size = config.QUERY_REPEAT_POOL_SIZE

    rng = random.Random(seed)
    unique_ratio = min(1.0, max(0.0, unique_ratio))
    unique_count = min(size, max(1, int(size * unique_ratio)))
    repeat_count = size - unique_count

    unique_queries = _generate_unique_queries(rng, unique_count)
    repeat_pool = unique_queries[: max(1, min(repeat_pool_size, len(unique_queries)))]
    workload = list(unique_queries)

    for _ in range(repeat_count):
        workload.append(rng.choice(repeat_pool))

    rng.shuffle(workload)
    return workload


QUERIES = generate_query_workload()
WARMUP_QUERIES = [
    query
    for queries in QUERY_TEMPLATES.values()
    for query in queries
]

# Wrapper (abstraction layer)
def send_to_scheduler(scheduler, request):
    return scheduler.handle_request(request)


# Send single request
def send_request(scheduler, request_id, query=None):
    if query is None:
        query = random.choice(QUERIES)
    request = Request(id=request_id, query=query)

    start = time.time()

    try:
        response = send_to_scheduler(scheduler, request)

        success = bool(getattr(response, "success", response.result != "FAILED"))
        latency = response.latency
        cached = bool(getattr(response, "cached", False))
        source = getattr(response, "source", "unknown")

    except Exception:
        success = False
        latency = time.time() - start
        cached = False
        source = "client_error"

    return {
        "request_id": request_id,
        "query": query,
        "success": success,
        "latency": latency,
        "cached": cached,
        "source": source,
    }


# Run load test
def run_client(scheduler, num_requests=None, queries=None):
    if num_requests is None:
        num_requests = config.NUM_USERS
    if queries is None:
        queries = QUERIES

    results = []
    unique_queries = len(set(queries[:num_requests]))
    unique_intents = len(
        {
            keyword_signature(query)
            for query in queries[:num_requests]
        }
    )

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = []

        for i in range(num_requests):
            query = queries[i % len(queries)]
            futures.append(executor.submit(send_request, scheduler, i, query))
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
    cache_hits = sum(1 for r in results if r.get("cached"))
    scheduler_cache_hits = sum(1 for r in results if r.get("source") == "scheduler_cache")
    llm_cache_hits = sum(1 for r in results if r.get("source") == "llm_cache")
    llm_inferences = sum(1 for r in results if r.get("source") == "llm_inference")

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
    print(f"Unique query strings: {unique_queries}")
    print(f"Unique query intents: {unique_intents}")
    print(f"Successful: {successes}")
    print(f"Failed: {failures}")
    print(f"Cache hits: {cache_hits}")
    print(f"Scheduler cache hits: {scheduler_cache_hits}")
    print(f"LLM cache hits: {llm_cache_hits}")
    print(f"LLM inferences: {llm_inferences}")
    print(f"Average latency: {avg_latency:.2f}s")
    print(f"P95 latency: {p95_latency:.2f}s")
    print(f"Throughput: {throughput:.2f} req/sec")
    print("================================\n")

    return {
        "summary": {
            "total_requests": total_requests,
            "unique_query_strings": unique_queries,
            "unique_query_intents": unique_intents,
            "successful": successes,
            "failed": failures,
            "cache_hits": cache_hits,
            "scheduler_cache_hits": scheduler_cache_hits,
            "llm_cache_hits": llm_cache_hits,
            "llm_inferences": llm_inferences,
            "average_latency": avg_latency,
            "p95_latency": p95_latency,
            "throughput": throughput,
            "total_time": total_time,
        },
        "requests": results,
    }
