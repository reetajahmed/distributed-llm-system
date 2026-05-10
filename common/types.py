from dataclasses import dataclass, field
import time
from typing import Optional


# Request sent from client to scheduler/worker.
@dataclass
class Request:
    id: int
    query: str
    timestamp: float = field(default_factory=time.time)


# Response returned by workers and caches.
@dataclass
class Response:
    id: int
    worker_id: int
    result: str
    latency: float
    strategy_used: Optional[str] = None
    success: bool = True
    cached: bool = False
    error: Optional[str] = None
    source: str = "worker"


# Minimal task state kept by the scheduler.
@dataclass
class Task:
    request: Request
    assigned_worker: Optional[int] = None
    start_time: float = field(default_factory=time.time)
    completed: bool = False
