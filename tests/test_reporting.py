# CSV report export tests.
import unittest
from argparse import Namespace
from collections import defaultdict
from unittest.mock import patch

from reporting import export_run_report


class FakeScheduler:
    completed = 10
    retried = 1
    retry_attempts = 2
    retry_successes = 1
    retry_failures = 0
    failed = 0
    cache_hits = 3
    source_counts = defaultdict(int, {"scheduler_cache": 3, "llm_inference": 7})


class TestReporting(unittest.TestCase):
    def test_export_run_report_writes_csv_files(self):
        args = Namespace(
            distributed=True,
            users=10,
            worker_urls=["http://w1", "http://w2"],
            workers=2,
            strategy="load_aware",
            query_seed=42,
            query_workload_size=10,
            query_unique_ratio=0.5,
            query_repeat_pool_size=2,
            warm_cache=False,
        )
        client_report = {
            "summary": {
                "total_requests": 10,
                "unique_query_strings": 7,
                "unique_query_intents": 7,
                "successful": 10,
                "failed": 0,
                "cache_hits": 3,
                "scheduler_cache_hits": 3,
                "llm_cache_hits": 0,
                "llm_inferences": 7,
                "average_latency": 0.1,
                "p95_latency": 0.2,
                "throughput": 20,
                "total_time": 0.5,
            },
            "requests": [
                {
                    "request_id": 1,
                    "query": "What is RAG?",
                    "success": True,
                    "latency": 0.1,
                    "cached": False,
                    "source": "llm_inference",
                }
            ],
        }
        lb_metrics = {
            "total_requests": 7,
            "failed_requests": 0,
            "logical_failed_requests": 0,
            "exception_failed_requests": 0,
            "failure_rate": 0,
            "average_latency": 0.2,
            "throughput": 10,
            "requests_per_worker": {1: 4, 2: 3},
            "strategy_counts": {"load_aware": 7},
            "source_counts": {"llm_inference": 7},
            "healthy_workers": 2,
        }
        worker_stats = {
            1: {"processed": 4, "source_counts": {"llm_inference": 4}},
            2: {"processed": 3, "source_counts": {"llm_inference": 3}},
        }

        with patch("pathlib.Path.mkdir"), patch("reporting._write_rows") as write_rows:
            export_dir = export_run_report(
                export_dir="reports",
                args=args,
                client_report=client_report,
                scheduler=FakeScheduler(),
                load_balancer_metrics=lb_metrics,
                worker_stats=worker_stats,
            )

        expected_files = {
            "run_summary.csv",
            "request_results.csv",
            "load_balancer_metrics.csv",
            "load_balancer_source_counts.csv",
            "requests_per_worker.csv",
            "scheduler_source_counts.csv",
            "worker_stats.csv",
        }
        written_files = {call.args[0].name for call in write_rows.call_args_list}

        self.assertTrue(expected_files.issubset(written_files))
        self.assertEqual(export_dir.parent.name, "reports")


if __name__ == "__main__":
    unittest.main()
