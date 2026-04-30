from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from common.types import Request

if TYPE_CHECKING:
    from workers.gpu_worker import GPUWorker


DEFAULT_GPU_CAPACITY = {
    1: 1.0,  # 100%
    2: 0.7,  # 70%
    3: 0.4,  # 40%
}


def _worker_id(worker: "GPUWorker") -> int:
    return int(getattr(worker, "id", getattr(worker, "worker_id", id(worker))))


def _is_healthy(worker: "GPUWorker") -> bool:
    health_check = getattr(worker, "is_healthy", None)
    if callable(health_check):
        return bool(health_check())

    if hasattr(worker, "healthy"):
        return bool(getattr(worker, "healthy"))

    return bool(getattr(worker, "alive", True))


def _active_connections(worker: "GPUWorker") -> int:
    return max(0, int(getattr(worker, "active_connections", 0)))


def _queue_length(worker: "GPUWorker") -> int:
    if hasattr(worker, "queue_length"):
        return max(0, int(getattr(worker, "queue_length")))

    queue = getattr(worker, "queue", None)
    if queue is not None:
        qsize = getattr(queue, "qsize", None)
        if callable(qsize):
            return max(0, int(qsize()))
        try:
            return max(0, len(queue))
        except TypeError:
            pass

    return max(0, int(getattr(worker, "pending_requests", 0)))


def _gpu_capacity(worker: "GPUWorker") -> float:
    capacity = getattr(worker, "gpu_capacity", None)
    if capacity is None:
        capacity = getattr(worker, "capacity", None)
    if capacity is None:
        capacity = getattr(worker, "weight", None)
    if capacity is None:
        capacity = DEFAULT_GPU_CAPACITY.get(_worker_id(worker), 1.0)

    capacity = float(capacity)
    if capacity > 1.0:
        capacity = capacity / 100.0

    return max(0.1, capacity)


def _rolling_latency(worker: "GPUWorker") -> float:
    for attr in ("rolling_latency", "average_latency", "avg_latency", "latency"):
        value = getattr(worker, attr, None)
        if value is not None:
            return max(0.0, float(value))
    return 0.0


def _healthy_workers(workers: List["GPUWorker"]) -> List["GPUWorker"]:
    return [worker for worker in workers if _is_healthy(worker)]


class RoundRobinBalancer:
    """Health-aware weighted round robin."""

    def __init__(self, workers: List["GPUWorker"]):
        self._workers = workers
        self._current_weights: Dict[int, float] = defaultdict(float)
        self._lock = threading.Lock()

    def get_worker(self) -> Optional["GPUWorker"]:
        with self._lock:
            healthy = _healthy_workers(self._workers)
            if not healthy:
                return None

            total_weight = sum(_gpu_capacity(worker) for worker in healthy)
            selected = healthy[0]
            selected_id = _worker_id(selected)

            for worker in healthy:
                worker_id = _worker_id(worker)
                self._current_weights[worker_id] += _gpu_capacity(worker)
                if self._current_weights[worker_id] > self._current_weights[selected_id]:
                    selected = worker
                    selected_id = worker_id

            self._current_weights[selected_id] -= total_weight
            return selected

    @property
    def name(self) -> str:
        return "round_robin"


class LeastConnectionsBalancer:
    """Routes to the healthy worker with the smallest capacity-adjusted load."""

    def __init__(self, workers: List["GPUWorker"]):
        self._workers = workers

    def get_worker(self) -> Optional["GPUWorker"]:
        healthy = _healthy_workers(self._workers)
        if not healthy:
            return None

        return min(
            healthy,
            key=lambda worker: (
                (_active_connections(worker) + _queue_length(worker)) / _gpu_capacity(worker),
                _active_connections(worker),
                _queue_length(worker),
                _worker_id(worker),
            ),
        )

    @property
    def name(self) -> str:
        return "least_connections"


class LoadAwareBalancer:
    """Routes by health, active work, queue depth, GPU capacity, and latency."""

    def __init__(self, workers: List["GPUWorker"]):
        self._workers = workers

    def _score(self, worker: "GPUWorker") -> float:
        active = _active_connections(worker)
        queued = _queue_length(worker)
        capacity = _gpu_capacity(worker)
        latency_penalty = _rolling_latency(worker) * 0.25
        return (active + (queued * 0.75) + latency_penalty) / capacity

    def get_worker(self) -> Optional["GPUWorker"]:
        healthy = _healthy_workers(self._workers)
        if not healthy:
            return None

        return min(
            healthy,
            key=lambda worker: (
                self._score(worker),
                _active_connections(worker),
                _queue_length(worker),
                _worker_id(worker),
            ),
        )

    @property
    def name(self) -> str:
        return "load_aware"


class LoadBalancer:
    """Aggregates routing strategies and tracks routing metrics."""

    STRATEGIES = ("round_robin", "least_connections", "load_aware")

    def __init__(self, workers: List["GPUWorker"], strategy: str = "load_aware"):
        self._workers = list(workers)
        self._round_robin = RoundRobinBalancer(self._workers)
        self._least_conn = LeastConnectionsBalancer(self._workers)
        self._load_aware = LoadAwareBalancer(self._workers)
        self._lock = threading.Lock()
        self._metrics: Dict[str, Any] = {
            "total_requests": 0,
            "failed_requests": 0,
            "total_latency": 0.0,
            "requests_per_worker": defaultdict(int),
            "strategy_counts": defaultdict(int),
        }
        self._strategy = "load_aware"
        self.set_strategy(strategy)

    @property
    def strategy(self) -> str:
        return self._strategy

    @strategy.setter
    def strategy(self, value: str) -> None:
        self.set_strategy(value)

    def set_strategy(self, strategy: str) -> None:
        if strategy not in self.STRATEGIES:
            valid = ", ".join(self.STRATEGIES)
            raise ValueError(f"Unknown load balancing strategy '{strategy}'. Valid strategies: {valid}")
        with self._lock:
            self._strategy = strategy

    def _pick(self) -> Optional["GPUWorker"]:
        strategy = self.strategy
        if strategy == "round_robin":
            return self._round_robin.get_worker()
        if strategy == "least_connections":
            return self._least_conn.get_worker()
        return self._load_aware.get_worker()

    def dispatch(self, request: Request) -> Any:
        start = time.time()
        worker = self._pick()
        if worker is None:
            self._record_metrics(None, self.strategy, 0.0, failed=True)
            raise RuntimeError("No healthy GPU workers available for routing")

        worker_id = _worker_id(worker)
        try:
            response = worker.process(request)
            if isinstance(response, dict):
                response["strategy_used"] = self.strategy
                response["worker_id"] = response.get("worker_id", worker_id)
                response["worker_capacity"] = _gpu_capacity(worker)
            else:
                setattr(response, "strategy_used", self.strategy)
                setattr(response, "worker_capacity", _gpu_capacity(worker))
            self._record_metrics(worker_id, self.strategy, time.time() - start, failed=False)
            return response
        except Exception:
            self._record_metrics(worker_id, self.strategy, time.time() - start, failed=True)
            raise

    def add_worker(self, worker: "GPUWorker") -> None:
        with self._lock:
            self._workers.append(worker)

    def remove_worker(self, worker_id: int) -> None:
        with self._lock:
            self._workers[:] = [worker for worker in self._workers if _worker_id(worker) != worker_id]

    def _record_metrics(
        self,
        worker_id: Optional[int],
        strategy: str,
        latency: float,
        failed: bool,
    ) -> None:
        with self._lock:
            self._metrics["total_requests"] += 1
            self._metrics["strategy_counts"][strategy] += 1
            self._metrics["total_latency"] += latency
            if worker_id is not None:
                self._metrics["requests_per_worker"][worker_id] += 1
            if failed:
                self._metrics["failed_requests"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = self._metrics["total_requests"]
            failures = self._metrics["failed_requests"]
            average_latency = self._metrics["total_latency"] / total if total else 0.0
            failure_rate = failures / total if total else 0.0

            return {
                "total_requests": total,
                "failed_requests": failures,
                "failure_rate": failure_rate,
                "average_latency": average_latency,
                "throughput": total / self._metrics["total_latency"] if self._metrics["total_latency"] else 0.0,
                "requests_per_worker": dict(self._metrics["requests_per_worker"]),
                "strategy_counts": dict(self._metrics["strategy_counts"]),
                "healthy_workers": self.alive_count,
            }

    @property
    def alive_count(self) -> int:
        return len(_healthy_workers(self._workers))
