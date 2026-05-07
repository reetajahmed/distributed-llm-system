import csv
import json
from datetime import datetime
from pathlib import Path


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_rows(path: Path, rows: list[dict], fieldnames: list[str] = None):
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _metric_rows(metrics: dict, prefix: str = ""):
    rows = []

    for key, value in metrics.items():
        metric = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(_metric_rows(value, prefix=f"{metric}."))
        else:
            rows.append({"metric": metric, "value": value})

    return rows


def _source_rows(source_counts: dict):
    return [
        {"source": source, "count": count}
        for source, count in sorted(source_counts.items())
    ]


def _requests_per_worker_rows(requests_per_worker: dict):
    return [
        {"worker_id": worker_id, "requests": count}
        for worker_id, count in sorted(requests_per_worker.items())
    ]


def _worker_rows(worker_stats: dict):
    rows = []

    for worker_id, stats in sorted(worker_stats.items()):
        row = {"worker_id": worker_id}
        for key, value in stats.items():
            if isinstance(value, dict):
                row[key] = json.dumps(value, sort_keys=True)
            else:
                row[key] = value
        rows.append(row)

    return rows


def export_run_report(
    export_dir,
    args,
    client_report,
    scheduler,
    load_balancer_metrics,
    worker_stats,
):
    run_id = _timestamp()
    base_dir = Path(export_dir) / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    client_summary = client_report["summary"]
    run_summary = {
        "run_id": run_id,
        "mode": "distributed" if args.distributed else "local",
        "users": args.users,
        "workers": len(args.worker_urls) if args.distributed else args.workers,
        "strategy": args.strategy,
        "query_seed": args.query_seed,
        "query_workload_size": args.query_workload_size,
        "query_unique_ratio": args.query_unique_ratio,
        "query_repeat_pool_size": args.query_repeat_pool_size,
        "warm_cache": args.warm_cache,
        **client_summary,
        "scheduler_tasks_completed": scheduler.completed,
        "scheduler_tasks_retried": scheduler.retried,
        "scheduler_retry_attempts": scheduler.retry_attempts,
        "scheduler_retry_successes": scheduler.retry_successes,
        "scheduler_retry_failures": scheduler.retry_failures,
        "scheduler_tasks_failed": scheduler.failed,
        "scheduler_cache_hits_summary": scheduler.cache_hits,
        "lb_total_routed_requests": load_balancer_metrics["total_requests"],
        "lb_failed_routed_requests": load_balancer_metrics["failed_requests"],
        "lb_logical_failed_routed_requests": load_balancer_metrics["logical_failed_requests"],
        "lb_exception_failed_routed_requests": load_balancer_metrics["exception_failed_requests"],
        "lb_failure_rate": load_balancer_metrics["failure_rate"],
        "lb_average_routing_latency": load_balancer_metrics["average_latency"],
        "lb_healthy_workers": load_balancer_metrics["healthy_workers"],
    }

    _write_rows(base_dir / "run_summary.csv", [run_summary])
    _write_rows(base_dir / "request_results.csv", client_report["requests"])
    _write_rows(base_dir / "load_balancer_metrics.csv", _metric_rows(load_balancer_metrics))
    _write_rows(
        base_dir / "load_balancer_source_counts.csv",
        _source_rows(load_balancer_metrics.get("source_counts", {})),
    )
    _write_rows(
        base_dir / "requests_per_worker.csv",
        _requests_per_worker_rows(load_balancer_metrics.get("requests_per_worker", {})),
    )
    _write_rows(
        base_dir / "scheduler_source_counts.csv",
        _source_rows(dict(scheduler.source_counts)),
    )
    _write_rows(base_dir / "worker_stats.csv", _worker_rows(worker_stats))

    return base_dir
