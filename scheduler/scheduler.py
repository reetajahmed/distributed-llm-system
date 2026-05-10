import copy
import time
import threading
from collections import OrderedDict, defaultdict
from dataclasses import replace

from common.text_utils import keyword_signature, normalize_text
from config import (
    SCHEDULER_CACHE_SIMILARITY_THRESHOLD,
    SCHEDULER_RESPONSE_CACHE_MAX_SIZE,
)
from scheduler.fault_tolerance import FaultTolerance


def _response_succeeded(response):
    # Accept both dataclass Response objects and dict fallback responses.
    if isinstance(response, dict):
        return response.get("success", False)

    if hasattr(response, "success"):
        return bool(getattr(response, "success"))

    return getattr(response, "result", None) is not None


def _response_strategy(response):
    if isinstance(response, dict):
        return response.get("strategy_used", "unknown")

    return getattr(response, "strategy_used", "unknown")


def _response_source(response):
    if isinstance(response, dict):
        return response.get("source", "unknown")
    return getattr(response, "source", "unknown")


def _response_retry_attempts(response):
    if isinstance(response, dict):
        return int(response.get("retry_attempts", 0))
    return int(getattr(response, "retry_attempts", 0))


class Scheduler:
    def __init__(self, load_balancer):
        # Scheduler coordinates cache hits, routing, retries, and counters.
        self.lb = load_balancer
        self.fault_handler = FaultTolerance(self.lb)

        # Tracking
        self.active_tasks = {}   # {request_id: status}
        self.results = {}        # {request_id: response}

        # Stats(for dashboard)
        self.completed = 0
        self.failed = 0
        self.retried = 0
        self.cache_hits = 0
        self.retry_attempts = 0
        self.retry_successes = 0
        self.retry_failures = 0
        self.source_counts = defaultdict(int)

        self._response_cache = OrderedDict()
        self._inflight = {}
        self._cache_lock = threading.RLock()

    def _cache_key(self, query: str) -> str:
        # Similar questions share a scheduler-level response cache key.
        signature = keyword_signature(query)
        if signature:
            return f"intent::{signature}"
        return f"query::{normalize_text(query)}"

    def _cache_key_tokens(self, cache_key: str) -> set[str]:
        _, _, value = cache_key.partition("::")
        return set(value.split())

    def _similarity(self, left_key: str, right_key: str) -> float:
        left = self._cache_key_tokens(left_key)
        right = self._cache_key_tokens(right_key)
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _find_similar_cache_key(self, cache_key: str):
        # Fuzzy cache lookup catches wording changes with same intent.
        best_key = None
        best_score = 0.0

        for existing_key in self._response_cache:
            score = self._similarity(cache_key, existing_key)
            if score > best_score:
                best_key = existing_key
                best_score = score

        if best_score >= SCHEDULER_CACHE_SIMILARITY_THRESHOLD:
            return best_key

        return None

    def _clone_cached_response(self, request, cached_response, latency: float):
        # Cached responses are copied so each request keeps its own id/latency.
        if isinstance(cached_response, dict):
            response = copy.deepcopy(cached_response)
            response["request_id"] = request.id
            response["id"] = request.id
            response["latency"] = latency
            response["cached"] = True
            response["strategy_used"] = "scheduler_cache"
            response["source"] = "scheduler_cache"
            return response

        response = replace(
            cached_response,
            id=request.id,
            latency=latency,
            cached=True,
        )
        response.strategy_used = "scheduler_cache"
        response.source = "scheduler_cache"
        return response

    def _get_cached_response(self, request, start_time: float):
        # Fast path: return a scheduler cache hit before touching workers.
        cache_key = self._cache_key(request.query)

        with self._cache_lock:
            cached_response = self._response_cache.get(cache_key)
            if cached_response is None:
                similar_key = self._find_similar_cache_key(cache_key)
                if similar_key is not None:
                    cache_key = similar_key
                    cached_response = self._response_cache.get(cache_key)

            if cached_response is None:
                return None

            self._response_cache.move_to_end(cache_key)
            self.cache_hits += 1

        latency = time.time() - start_time
        response = self._clone_cached_response(request, cached_response, latency)
        print(
            f"[Scheduler] Response cache hit for request {request.id} "
            f"| source=scheduler_cache | latency={latency:.4f}s"
        )
        return response

    def _record_source(self, response):
        self.source_counts[_response_source(response)] += 1

    def _store_cached_response(self, request, response):
        # Only successful worker answers are reusable.
        if not _response_succeeded(response):
            return

        cache_key = self._cache_key(request.query)

        with self._cache_lock:
            self._response_cache[cache_key] = copy.deepcopy(response)
            self._response_cache.move_to_end(cache_key)
            print(
                f"[Scheduler] Stored response cache for request {request.id} "
                f"| key={cache_key}"
            )

            while len(self._response_cache) > SCHEDULER_RESPONSE_CACHE_MAX_SIZE:
                self._response_cache.popitem(last=False)

    def _wait_for_inflight(self, cache_key: str):
        # Prevent duplicate in-flight requests from computing the same answer.
        with self._cache_lock:
            event = self._inflight.get(cache_key)
            if event is None:
                event = threading.Event()
                self._inflight[cache_key] = event
                return event, True

        event.wait()
        return event, False

    def _finish_inflight(self, cache_key: str):
        with self._cache_lock:
            event = self._inflight.pop(cache_key, None)
            if event is not None:
                event.set()

    def handle_request(self, request):
        # Main request lifecycle: cache, dispatch, retry, record result.
        print(f"[Scheduler] Received request {request.id}")

        start_time = time.time()
        cache_key = self._cache_key(request.query)

        cached_response = self._get_cached_response(request, start_time)
        if cached_response is not None:
            self.active_tasks[request.id] = "CACHE_HIT"
            self.results[request.id] = cached_response
            self.completed += 1
            self._record_source(cached_response)
            return cached_response

        _, owns_inflight = self._wait_for_inflight(cache_key)

        if not owns_inflight:
            cached_response = self._get_cached_response(request, start_time)
            if cached_response is not None:
                self.active_tasks[request.id] = "CACHE_HIT"
                self.results[request.id] = cached_response
                self.completed += 1
                self._record_source(cached_response)
                return cached_response

        self.active_tasks[request.id] = "RUNNING"

        try:
            response = self.lb.dispatch(request)

            success = _response_succeeded(response)
            strategy = _response_strategy(response)

            print(f"[Scheduler] Strategy used: {strategy}")

            if success:
                self.active_tasks[request.id] = "COMPLETED"
                self.completed += 1
            else:
                raise Exception("Worker returned unsuccessful result")

            self.results[request.id] = response
            self._store_cached_response(request, response)
            self._record_source(response)

            latency = time.time() - start_time
            source = _response_source(response)
            print(
                f"[Scheduler] Request {request.id} completed "
                f"| source={source} | latency={latency:.3f}s"
            )

            return response

        except Exception as e:
            print(f"[Scheduler] ERROR on request {request.id}: {e}")

            # Failed dispatches go through retry/reassignment logic.
            self.active_tasks[request.id] = "RETRYING"
            self.retried += 1

            response = self.fault_handler.handle_failure(request)


            success = _response_succeeded(response)
            retry_attempts = _response_retry_attempts(response)
            self.retry_attempts += retry_attempts

            if success:
                self.active_tasks[request.id] = "REASSIGNED"
                self.completed += 1
                self.retry_successes += 1
                self.results[request.id] = response
                self._store_cached_response(request, response)
                self._record_source(response)

                print(f"[Recovery] Request {request.id} reassigned successfully")
            else:
                self.active_tasks[request.id] = "FAILED"
                self.failed += 1
                self.retry_failures += 1
                self._record_source(response)

            return response
        finally:
            if owns_inflight:
                self._finish_inflight(cache_key)

    def print_status(self):
        print("\n===== Scheduler Status =====")
        for req_id, status in self.active_tasks.items():
            print(f"Request {req_id}: {status}")
        print("============================\n")

    def print_summary(self):
        print("\n===== Scheduler Summary =====")
        print(f"Tasks completed: {self.completed}")
        print(f"Tasks retried: {self.retried}")
        print(f"Retry attempts: {self.retry_attempts}")
        print(f"Retry successes: {self.retry_successes}")
        print(f"Retry failures: {self.retry_failures}")
        print(f"Tasks failed: {self.failed}")
        print(f"Scheduler cache hits: {self.cache_hits}")
        print(f"Source counts: {dict(self.source_counts)}")
        print("============================\n")

    def clear_cache(self):
        # Useful for tests and repeated experiments in one process.
        with self._cache_lock:
            self._response_cache.clear()
            self._inflight.clear()
