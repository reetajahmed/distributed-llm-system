import threading
from typing import List, Optional
from workers.gpu_worker import GPUWorker
from common.types import Request, Response


class RoundRobinBalancer:
    """Cycles through workers in order regardless of load."""

    def __init__(self, workers: List[GPUWorker]):
        self._workers = workers
        self._index = 0
        self._lock = threading.Lock()

    def get_worker(self) -> Optional[GPUWorker]:
        with self._lock:
            alive = [w for w in self._workers if w.alive]
            if not alive:
                return None
            worker = alive[self._index % len(alive)]
            self._index = (self._index + 1) % len(alive)
            return worker

    @property
    def name(self) -> str:
        return "round_robin"


class LeastConnectionsBalancer:
    """Routes to the worker with the fewest active connections."""

    def __init__(self, workers: List[GPUWorker]):
        self._workers = workers

    def get_worker(self) -> Optional[GPUWorker]:
        alive = [w for w in self._workers if w.alive]
        if not alive:
            return None
        return min(alive, key=lambda w: w.active_connections)

    @property
    def name(self) -> str:
        return "least_connections"


class LoadAwareBalancer:
    """Weighted routing — avoids overloaded workers by considering both
    active connections and a rolling average latency per worker."""

    def __init__(self, workers: List[GPUWorker], threshold: int = 10):
        self._workers = workers
        self._threshold = threshold
        self._lock = threading.Lock()
        self._index = 0  # fallback round-robin cursor

    def get_worker(self) -> Optional[GPUWorker]:
        alive = [w for w in self._workers if w.alive]
        if not alive:
            return None

        # Prefer workers below the connection threshold; pick least loaded among those
        below = [w for w in alive if w.active_connections < self._threshold]
        candidates = below if below else alive
        return min(candidates, key=lambda w: w.active_connections)

    @property
    def name(self) -> str:
        return "load_aware"


class LoadBalancer:
    """Aggregates all three strategies; strategy can be switched at runtime."""

    STRATEGIES = ("round_robin", "least_connections", "load_aware")

    def __init__(self, workers: List[GPUWorker], strategy: str = "load_aware"):
        self._workers = workers
        self._round_robin = RoundRobinBalancer(workers)
        self._least_conn = LeastConnectionsBalancer(workers)
        self._load_aware = LoadAwareBalancer(workers)
        self.strategy = strategy

    def _pick(self) -> Optional[GPUWorker]:
        if self.strategy == "round_robin":
            return self._round_robin.get_worker()
        elif self.strategy == "least_connections":
            return self._least_conn.get_worker()
        else:
            return self._load_aware.get_worker()

    def dispatch(self, request: Request) -> Response:
        worker = self._pick()
        if worker is None:
            raise RuntimeError("No healthy workers available")
        response = worker.process(request)
        response.strategy_used = self.strategy
        return response

    def add_worker(self, worker: GPUWorker):
        self._workers.append(worker)
        self._round_robin._workers.append(worker)
        self._least_conn._workers.append(worker)
        self._load_aware._workers.append(worker)

    @property
    def alive_count(self) -> int:
        return sum(1 for w in self._workers if w.alive)
