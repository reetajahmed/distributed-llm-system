# Load balancer strategy, health, and metrics tests.
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.types import Request
from load_balancer.load_balancer import LoadBalancer


class FakeWorker:
    def __init__(
        self,
        worker_id,
        alive=True,
        gpu_capacity=None,
        active_connections=0,
        queue_length=0,
        rolling_latency=0.0,
        fail=False,
        success=True,
        source="worker",
    ):
        self.id = worker_id
        self.alive = alive
        self.active_connections = active_connections
        self.queue_length = queue_length
        self.rolling_latency = rolling_latency
        self.fail = fail
        self.success = success
        self.source = source

        if gpu_capacity is not None:
            self.gpu_capacity = gpu_capacity

    def process(self, request):
        if self.fail:
            raise RuntimeError("worker failed")

        return {
            "request_id": request.id,
            "answer": "ok",
            "success": self.success,
            "latency": 0.01,
            "source": self.source,
        }


class TestLoadBalancer(unittest.TestCase):
    def test_invalid_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown load balancing strategy"):
            LoadBalancer([], strategy="random")

    def test_dispatch_skips_unhealthy_workers(self):
        workers = [
            FakeWorker(1, alive=False, gpu_capacity=1.0),
            FakeWorker(2, alive=True, gpu_capacity=0.7),
        ]

        lb = LoadBalancer(workers, strategy="load_aware")
        response = lb.dispatch(Request(1, "test query"))

        self.assertEqual(response["worker_id"], 2)
        self.assertEqual(response["strategy_used"], "load_aware")
        self.assertEqual(response["worker_capacity"], 0.7)

    def test_no_healthy_workers_raises_clear_error(self):
        lb = LoadBalancer([FakeWorker(1, alive=False)], strategy="load_aware")

        with self.assertRaisesRegex(RuntimeError, "No healthy GPU workers"):
            lb.dispatch(Request(1, "test query"))

        metrics = lb.get_metrics()
        self.assertEqual(metrics["total_requests"], 1)
        self.assertEqual(metrics["failed_requests"], 1)
        self.assertEqual(metrics["exception_failed_requests"], 1)
        self.assertEqual(metrics["failure_rate"], 1.0)

    def test_weighted_round_robin_favors_higher_capacity_worker(self):
        workers = [
            FakeWorker(1, gpu_capacity=1.0),
            FakeWorker(2, gpu_capacity=0.4),
        ]

        lb = LoadBalancer(workers, strategy="round_robin")

        selected_workers = [
            lb.dispatch(Request(request_id, "test query"))["worker_id"]
            for request_id in range(1, 8)
        ]

        self.assertGreater(selected_workers.count(1), selected_workers.count(2))
        self.assertEqual(set(selected_workers), {1, 2})

    def test_load_aware_considers_queue_length_and_capacity(self):
        workers = [
            FakeWorker(1, gpu_capacity=1.0, active_connections=0, queue_length=5),
            FakeWorker(2, gpu_capacity=0.7, active_connections=0, queue_length=0),
        ]

        lb = LoadBalancer(workers, strategy="load_aware")
        response = lb.dispatch(Request(1, "test query"))

        self.assertEqual(response["worker_id"], 2)

    def test_least_connections_uses_capacity_adjusted_load(self):
        workers = [
            FakeWorker(1, gpu_capacity=1.0, active_connections=2, queue_length=0),
            FakeWorker(2, gpu_capacity=0.4, active_connections=1, queue_length=0),
        ]

        lb = LoadBalancer(workers, strategy="least_connections")
        response = lb.dispatch(Request(1, "test query"))

        self.assertEqual(response["worker_id"], 1)

    def test_strategy_can_switch_at_runtime_and_metrics_are_recorded(self):
        workers = [
            FakeWorker(1, gpu_capacity=1.0),
            FakeWorker(2, gpu_capacity=0.7),
        ]

        lb = LoadBalancer(workers, strategy="round_robin")

        first = lb.dispatch(Request(1, "test query"))
        lb.strategy = "load_aware"
        second = lb.dispatch(Request(2, "test query"))
        metrics = lb.get_metrics()

        self.assertEqual(first["strategy_used"], "round_robin")
        self.assertEqual(second["strategy_used"], "load_aware")
        self.assertEqual(metrics["total_requests"], 2)
        self.assertEqual(metrics["failed_requests"], 0)
        self.assertEqual(metrics["strategy_counts"]["round_robin"], 1)
        self.assertEqual(metrics["strategy_counts"]["load_aware"], 1)
        self.assertEqual(sum(metrics["requests_per_worker"].values()), 2)
        self.assertEqual(metrics["source_counts"]["worker"], 2)

    def test_unsuccessful_response_updates_logical_failure_metrics(self):
        lb = LoadBalancer(
            [FakeWorker(1, alive=True, success=False, source="llm_timeout")],
            strategy="load_aware",
        )

        response = lb.dispatch(Request(1, "test query"))
        metrics = lb.get_metrics()

        self.assertFalse(response["success"])
        self.assertEqual(metrics["total_requests"], 1)
        self.assertEqual(metrics["failed_requests"], 1)
        self.assertEqual(metrics["logical_failed_requests"], 1)
        self.assertEqual(metrics["exception_failed_requests"], 0)
        self.assertEqual(metrics["source_counts"]["llm_timeout"], 1)

    def test_worker_process_failure_updates_metrics(self):
        lb = LoadBalancer(
            [FakeWorker(1, alive=True, fail=True)],
            strategy="load_aware",
        )

        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            lb.dispatch(Request(1, "test query"))

        metrics = lb.get_metrics()
        self.assertEqual(metrics["total_requests"], 1)
        self.assertEqual(metrics["failed_requests"], 1)
        self.assertEqual(metrics["requests_per_worker"][1], 1)

    def test_health_method_takes_priority_over_alive_flag(self):
        class HealthCheckedWorker(FakeWorker):
            def is_healthy(self):
                return False

        workers = [
            HealthCheckedWorker(1, alive=True, gpu_capacity=1.0),
            FakeWorker(2, alive=True, gpu_capacity=0.7),
        ]

        lb = LoadBalancer(workers, strategy="load_aware")
        response = lb.dispatch(Request(1, "test query"))

        self.assertEqual(response["worker_id"], 2)

    def test_capacity_can_be_percentage_value(self):
        workers = [
            FakeWorker(1, gpu_capacity=100, active_connections=1),
            FakeWorker(2, gpu_capacity=40, active_connections=1),
        ]

        lb = LoadBalancer(workers, strategy="least_connections")
        response = lb.dispatch(Request(1, "test query"))

        self.assertEqual(response["worker_id"], 1)


if __name__ == "__main__":
    unittest.main()
