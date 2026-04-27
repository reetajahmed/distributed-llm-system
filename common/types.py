from dataclasses import dataclass, field
import time
from typing import Optional


# Request Object
@dataclass
class Request:
    id: int
    query: str
    timestamp: float = field(default_factory=time.time)


# Response Object
@dataclass
class Response:
    id: int
    worker_id: int
    result: str
    latency: float


# Task Tracking (for Scheduler)
class Task:
    request: Request
    assigned_worker: Optional[int] = None
    start_time: float = field(default_factory=time.time)
    completed: bool = False