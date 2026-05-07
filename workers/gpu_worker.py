import time
import threading
from collections import defaultdict

from llm.model import run_llm_with_metrics
from workers.rag import retrieve_context
from common.types import Response


class GPUWorker:
    def __init__(self, worker_id: int):
        self.id = worker_id
        self.alive = True                 
        self.active_connections = 0
        self._stats_lock = threading.Lock()
        self._stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "cache_hits": 0,
            "llm_cache_hits": 0,
            "llm_inferences": 0,
            "timeouts": 0,
            "errors": 0,
            "total_latency": 0.0,
            "source_counts": defaultdict(int),
        }

    def process(self, request):
        start = time.time()
        self.active_connections += 1     

        print(f"[Worker {self.id}] Processing request {request.id}")

        try:
            context = retrieve_context(request.query)
            llm_result = run_llm_with_metrics(
                request.query,
                context,
                request_id=request.id,
            )
            result = llm_result["answer"]
            success = llm_result["success"]
            cached = llm_result["cached"]
            error = llm_result["error"]
            source = llm_result.get("source", "llm_cache" if cached else "llm_inference")

            print(
                f"[Worker {self.id}] Request {request.id} answer source={source} "
                f"| success={success}"
            )

        except Exception as e:
            print(f"[Worker {self.id}] ERROR: {e}")
            result = f"Worker failed safely: {str(e)}"
            success = False
            cached = False
            error = str(e)
            source = "worker_error"

        finally:
            self.active_connections -= 1  

        latency = time.time() - start

        response = Response(
            id=request.id,
            worker_id=self.id,
            result=result,
            latency=latency,
            success=success,
            cached=cached,
            error=error,
            source=source,
        )

        print(
            f"[Worker {self.id}] Finished request {request.id} "
            f"| source={source} | latency={latency:.3f}s"
        )

        self._record_stats(success, cached, source, error, latency)

        return response

    def _record_stats(self, success, cached, source, error, latency):
        with self._stats_lock:
            self._stats["processed"] += 1
            self._stats["total_latency"] += latency
            self._stats["source_counts"][source] += 1

            if success:
                self._stats["successful"] += 1
            else:
                self._stats["failed"] += 1

            if cached:
                self._stats["cache_hits"] += 1
            if source == "llm_cache":
                self._stats["llm_cache_hits"] += 1
            elif source == "llm_inference":
                self._stats["llm_inferences"] += 1
            elif source == "llm_timeout":
                self._stats["timeouts"] += 1
            elif source in ("llm_error", "worker_error") or error:
                self._stats["errors"] += 1

    def get_stats(self):
        with self._stats_lock:
            processed = self._stats["processed"]
            average_latency = (
                self._stats["total_latency"] / processed
                if processed
                else 0.0
            )

            return {
                "worker_id": self.id,
                "alive": self.alive,
                "active_connections": self.active_connections,
                "processed": processed,
                "successful": self._stats["successful"],
                "failed": self._stats["failed"],
                "cache_hits": self._stats["cache_hits"],
                "llm_cache_hits": self._stats["llm_cache_hits"],
                "llm_inferences": self._stats["llm_inferences"],
                "timeouts": self._stats["timeouts"],
                "errors": self._stats["errors"],
                "average_latency": average_latency,
                "source_counts": dict(self._stats["source_counts"]),
            }
