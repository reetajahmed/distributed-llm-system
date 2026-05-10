import argparse

import config
from client.client import WARMUP_QUERIES
from client.client import generate_query_workload
from client.client import run_client
from common.default_docs import DEFAULT_DOCUMENTS
from common.types import Request
from load_balancer.load_balancer import LoadBalancer
from reporting import export_run_report
from scheduler.scheduler import Scheduler
from workers.gpu_worker import GPUWorker
from workers.rag import ingest_documents
from workers.remote_worker import RemoteGPUWorker


def build_scheduler(
    strategy: str,
    worker_count: int,
    distributed: bool = False,
    worker_urls: list[str] = None,
) -> Scheduler:
    if distributed:
        if not worker_urls:
            raise ValueError("Distributed mode requires at least one worker URL.")

        workers = [
            RemoteGPUWorker(
                worker_id=index + 1,
                base_url=url,
                timeout_seconds=config.REMOTE_WORKER_TIMEOUT_SECONDS,
                gpu_capacity=config.LOAD_BALANCER_DEFAULT_GPU_CAPACITY.get(index + 1),
            )
            for index, url in enumerate(worker_urls)
        ]
    else:
        workers = [GPUWorker(worker_id=i + 1) for i in range(worker_count)]

    load_balancer = LoadBalancer(workers, strategy=strategy)
    return Scheduler(load_balancer)


def warm_cache(scheduler: Scheduler):
    print("\n===== Warming Cache =====")
    for index, query in enumerate(WARMUP_QUERIES):
        response = scheduler.handle_request(Request(id=-(index + 1), query=query))
        status = "cached" if response.cached else "computed"
        print(f"Warm query '{query}' {status} | success={response.success}")
    print("=========================\n")

    scheduler.active_tasks.clear()
    scheduler.results.clear()
    scheduler.completed = 0
    scheduler.failed = 0
    scheduler.retried = 0
    scheduler.cache_hits = 0
    scheduler.retry_attempts = 0
    scheduler.retry_successes = 0
    scheduler.retry_failures = 0
    scheduler.source_counts.clear()
    scheduler.lb.reset_metrics()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the distributed LLM system load test."
    )
    parser.add_argument(
        "--users",
        type=int,
        default=config.NUM_USERS,
        help="Number of simulated user requests to send.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=len(config.LOAD_BALANCER_DEFAULT_GPU_CAPACITY),
        help="Number of GPU workers to create.",
    )
    parser.add_argument(
        "--strategy",
        choices=config.LOAD_BALANCER_STRATEGIES,
        default=config.LOAD_BALANCER_DEFAULT_STRATEGY,
        help="Load-balancing strategy to use.",
    )
    parser.add_argument(
        "--skip-rag-ingest",
        action="store_true",
        help="Skip ingestion of the default RAG documents.",
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Precompute answers for every query template before the load test.",
    )
    parser.add_argument(
        "--query-seed",
        type=int,
        default=config.QUERY_RANDOM_SEED,
        help="Random seed for generated query workload.",
    )
    parser.add_argument(
        "--query-workload-size",
        type=int,
        default=config.QUERY_WORKLOAD_SIZE,
        help="Number of random queries to generate before the load test.",
    )
    parser.add_argument(
        "--query-unique-ratio",
        type=float,
        default=config.QUERY_UNIQUE_RATIO,
        help="Fraction of generated queries that should be unique.",
    )
    parser.add_argument(
        "--query-repeat-pool-size",
        type=int,
        default=config.QUERY_REPEAT_POOL_SIZE,
        help="Number of unique queries reused for repeated cache-hit traffic.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Route requests to remote worker HTTP services.",
    )
    parser.add_argument(
        "--worker-urls",
        default=",".join(config.DEFAULT_REMOTE_WORKER_URLS),
        help="Comma-separated remote worker URLs for distributed mode.",
    )
    parser.add_argument(
        "--export-dir",
        default=config.REPORT_EXPORT_DIR,
        help="Directory for CSV report exports. Use an empty value to disable.",
    )
    parser.add_argument(
        "--max-client-workers",
        type=int,
        default=config.MAX_WORKERS,
        help="Maximum concurrent client request threads.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=config.REQUEST_DELAY,
        help="Delay in seconds between submitted client requests.",
    )
    args = parser.parse_args()

    if args.users < 1:
        parser.error("--users must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_client_workers < 1:
        parser.error("--max-client-workers must be at least 1")
    if args.request_delay < 0:
        parser.error("--request-delay must be zero or greater")
    if args.query_workload_size < 1:
        parser.error("--query-workload-size must be at least 1")
    if not 0 <= args.query_unique_ratio <= 1:
        parser.error("--query-unique-ratio must be between 0 and 1")
    if args.query_repeat_pool_size < 1:
        parser.error("--query-repeat-pool-size must be at least 1")
    args.worker_urls = [
        url.strip()
        for url in args.worker_urls.split(",")
        if url.strip()
    ]
    if args.distributed and not args.worker_urls:
        parser.error("--distributed requires at least one --worker-urls value")

    return args


def main():
    args = parse_args()

    print("\n===== Distributed LLM System =====")
    print(f"Users: {args.users}")
    print(f"Workers: {args.workers}")
    print(f"Strategy: {args.strategy}")
    print(f"Mode: {'distributed HTTP workers' if args.distributed else 'local in-process workers'}")
    print(f"Query seed: {args.query_seed}")
    print(f"Generated query workload: {args.query_workload_size}")
    print(f"Query unique ratio: {args.query_unique_ratio:.2f}")
    print(f"Query repeat pool size: {args.query_repeat_pool_size}")
    print(f"Max client workers: {args.max_client_workers}")
    print(f"Request delay: {args.request_delay}")
    if args.distributed:
        print(f"Worker URLs: {args.worker_urls}")

    if args.distributed:
        print("RAG ingestion is handled by each worker server in distributed mode.")
    elif not args.skip_rag_ingest:
        ingest_result = ingest_documents(DEFAULT_DOCUMENTS)
        print(
            "RAG documents: "
            f"{ingest_result['total_documents']} total "
            f"({ingest_result['added']} added)"
        )

    scheduler = build_scheduler(
        strategy=args.strategy,
        worker_count=args.workers,
        distributed=args.distributed,
        worker_urls=args.worker_urls,
    )

    if args.warm_cache:
        warm_cache(scheduler)

    queries = generate_query_workload(
        size=args.query_workload_size,
        seed=args.query_seed,
        unique_ratio=args.query_unique_ratio,
        repeat_pool_size=args.query_repeat_pool_size,
    )

    client_report = run_client(
        scheduler,
        num_requests=args.users,
        queries=queries,
        max_workers=args.max_client_workers,
        request_delay=args.request_delay,
    )

    scheduler.print_summary()

    metrics = scheduler.lb.get_metrics()
    print("===== Load Balancer Metrics =====")
    print(f"Total routed requests: {metrics['total_requests']}")
    print(f"Failed routed requests: {metrics['failed_requests']}")
    print(f"Logical failed routed requests: {metrics['logical_failed_requests']}")
    print(f"Exception failed routed requests: {metrics['exception_failed_requests']}")
    print(f"Failure rate: {metrics['failure_rate']:.2%}")
    print(f"Average routing latency: {metrics['average_latency']:.3f}s")
    print(f"Healthy workers: {metrics['healthy_workers']}")
    print(f"Requests per worker: {metrics['requests_per_worker']}")
    print(f"Strategy counts: {metrics['strategy_counts']}")
    print(f"Source counts: {metrics['source_counts']}")
    print("=================================\n")

    worker_stats = scheduler.lb.get_worker_stats()
    if worker_stats:
        print("===== Worker Stats =====")
        for worker_id, stats in worker_stats.items():
            print(f"Worker {worker_id}: {stats}")
        print("========================\n")

    if args.export_dir:
        export_path = export_run_report(
            export_dir=args.export_dir,
            args=args,
            client_report=client_report,
            scheduler=scheduler,
            load_balancer_metrics=metrics,
            worker_stats=worker_stats,
        )
        print(f"CSV reports exported to: {export_path}")


if __name__ == "__main__":
    main()
