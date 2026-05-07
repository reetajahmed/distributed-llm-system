import unittest
from unittest.mock import Mock, patch

from common.types import Request
from workers.remote_worker import RemoteGPUWorker


class TestRemoteGPUWorker(unittest.TestCase):
    def test_process_posts_request_and_returns_response(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "id": 7,
            "worker_id": 2,
            "result": "ok",
            "latency": 0.25,
            "success": True,
            "cached": False,
            "error": None,
            "source": "llm_inference",
        }

        requests_module = Mock()
        requests_module.post.return_value = response
        requests_module.RequestException = Exception

        with patch("workers.remote_worker._requests", return_value=requests_module):
            worker = RemoteGPUWorker(2, "http://worker.local")
            result = worker.process(Request(7, "What is RAG?"))

        self.assertEqual(result.id, 7)
        self.assertEqual(result.worker_id, 2)
        self.assertEqual(result.result, "ok")
        self.assertTrue(result.success)
        self.assertEqual(result.source, "llm_inference")
        requests_module.post.assert_called_once()

    def test_health_false_when_worker_unreachable(self):
        requests_module = Mock()
        requests_module.get.side_effect = Exception("offline")
        requests_module.RequestException = Exception

        with patch("workers.remote_worker._requests", return_value=requests_module):
            worker = RemoteGPUWorker(1, "http://worker.local")
            healthy = worker.is_healthy()

        self.assertFalse(healthy)
        self.assertFalse(worker.alive)

    def test_get_stats_fetches_remote_stats(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "worker_id": 1,
            "processed": 3,
            "successful": 3,
        }

        requests_module = Mock()
        requests_module.get.return_value = response
        requests_module.RequestException = Exception

        with patch("workers.remote_worker._requests", return_value=requests_module):
            worker = RemoteGPUWorker(1, "http://worker.local")
            stats = worker.get_stats()

        self.assertEqual(stats["worker_id"], 1)
        self.assertEqual(stats["processed"], 3)
        self.assertEqual(stats["remote_url"], "http://worker.local")


if __name__ == "__main__":
    unittest.main()
