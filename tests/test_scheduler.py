import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.types import Request, Response
from load_balancer.load_balancer import LoadBalancer
from scheduler.scheduler import Scheduler


class ResponseWorker:
    def __init__(self, worker_id, fail=False):
        self.id = worker_id
        self.alive = True
        self.active_connections = 0
        self.fail = fail

    def process(self, request):
        if self.fail:
            raise RuntimeError(f"worker {self.id} failed")

        return Response(
            id=request.id,
            worker_id=self.id,
            result=f"answer from worker {self.id}",
            latency=0.01,
        )


class TestSchedulerWithProjectModules(unittest.TestCase):
    def test_successful_response_object_is_marked_completed(self):
        lb = LoadBalancer([ResponseWorker(1)], strategy="load_aware")
        scheduler = Scheduler(lb)

        response = scheduler.handle_request(Request(1, "What is RAG?"))

        self.assertIsInstance(response, Response)
        self.assertEqual(response.worker_id, 1)
        self.assertEqual(response.strategy_used, "load_aware")
        self.assertEqual(scheduler.active_tasks[1], "COMPLETED")
        self.assertEqual(scheduler.results[1], response)
        self.assertEqual(scheduler.completed, 1)
        self.assertEqual(scheduler.retried, 0)
        self.assertEqual(scheduler.failed, 0)

    def test_fault_tolerance_reassigns_after_worker_failure(self):
        workers = [
            ResponseWorker(1, fail=True),
            ResponseWorker(2),
        ]
        lb = LoadBalancer(workers, strategy="round_robin")
        scheduler = Scheduler(lb)

        response = scheduler.handle_request(Request(2, "Explain fault tolerance."))

        self.assertIsInstance(response, Response)
        self.assertEqual(response.worker_id, 2)
        self.assertEqual(response.strategy_used, "round_robin")
        self.assertEqual(scheduler.active_tasks[2], "REASSIGNED")
        self.assertEqual(scheduler.results[2], response)
        self.assertEqual(scheduler.completed, 1)
        self.assertEqual(scheduler.retried, 1)
        self.assertEqual(scheduler.failed, 0)

    def test_permanent_failure_returns_failure_payload(self):
        lb = LoadBalancer([ResponseWorker(1, fail=True)], strategy="load_aware")
        scheduler = Scheduler(lb)

        response = scheduler.handle_request(Request(3, "This will fail."))

        self.assertEqual(response["request_id"], 3)
        self.assertFalse(response["success"])
        self.assertEqual(response["answer"], "FAILED")
        self.assertEqual(scheduler.active_tasks[3], "FAILED")
        self.assertEqual(scheduler.completed, 0)
        self.assertEqual(scheduler.retried, 1)
        self.assertEqual(scheduler.failed, 1)


class TestSchedulerWithRealRag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from workers import rag
            from workers.gpu_worker import GPUWorker
        except Exception as exc:
            raise unittest.SkipTest(f"Real RAG dependencies are unavailable: {exc}")

        cls.rag = rag
        cls.GPUWorker = GPUWorker

    def setUp(self):
        self.rag.DOCUMENTS.clear()
        self.rag.DOC_IDS.clear()
        self.rag._SEEN_DOC_HASHES.clear()
        self.rag._cache.clear()
        self.rag._query_embedding_cache.clear()
        self.rag.index = None

        try:
            result = self.rag.ingest_documents(
                [
                    "RAG retrieves relevant documents before generating an answer.",
                    "Fault tolerance allows a system to continue working after worker failure.",
                    "Load balancing distributes incoming requests across multiple workers.",
                ]
            )
        except Exception as exc:
            raise unittest.SkipTest(f"Real RAG model is unavailable: {exc}")

        if not result["success"]:
            raise unittest.SkipTest(f"Real RAG ingestion failed: {result}")

    def test_scheduler_dispatches_to_gpu_worker_using_real_rag(self):
        worker = self.GPUWorker(worker_id=1)
        lb = LoadBalancer([worker], strategy="load_aware")
        scheduler = Scheduler(lb)

        def answer_with_context(query, context):
            return f"query={query}\ncontext={context}"

        with patch("workers.gpu_worker.run_llm", side_effect=answer_with_context):
            response = scheduler.handle_request(
                Request(
                    4,
                    "RAG retrieves relevant documents before generating an answer.",
                )
            )

        self.assertIsInstance(response, Response)
        self.assertEqual(response.worker_id, 1)
        self.assertEqual(response.strategy_used, "load_aware")
        self.assertIn("RAG retrieves relevant documents", response.result)
        self.assertEqual(scheduler.active_tasks[4], "COMPLETED")
        self.assertEqual(scheduler.completed, 1)
        self.assertEqual(scheduler.retried, 0)
        self.assertEqual(scheduler.failed, 0)


if __name__ == "__main__":
    unittest.main()
