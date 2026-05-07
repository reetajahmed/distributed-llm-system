import time

from common.types import Response


def _requests():
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Distributed mode requires the 'requests' package. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    return requests


class RemoteGPUWorker:
    def __init__(
        self,
        worker_id: int,
        base_url: str,
        timeout_seconds: float = 60.0,
        gpu_capacity: float = None,
    ):
        self.id = worker_id
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.active_connections = 0
        self.queue_length = 0
        self.rolling_latency = 0.0
        self.alive = True

        if gpu_capacity is not None:
            self.gpu_capacity = gpu_capacity

    def is_healthy(self) -> bool:
        requests = _requests()
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=min(3.0, self.timeout_seconds),
            )
            healthy = response.status_code == 200 and response.json().get("alive", False)
            self.alive = bool(healthy)
            return self.alive
        except requests.RequestException:
            self.alive = False
            return False

    def process(self, request):
        requests = _requests()
        start = time.time()
        self.active_connections += 1

        try:
            payload = {
                "id": request.id,
                "query": request.query,
                "timestamp": request.timestamp,
            }
            response = requests.post(
                f"{self.base_url}/process",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()

            latency = time.time() - start
            self.rolling_latency = (
                (self.rolling_latency * 0.8) + (latency * 0.2)
                if self.rolling_latency
                else latency
            )
            self.alive = True

            return Response(
                id=int(data.get("id", request.id)),
                worker_id=int(data.get("worker_id", self.id)),
                result=data.get("result", ""),
                latency=float(data.get("latency", latency)),
                success=bool(data.get("success", True)),
                cached=bool(data.get("cached", False)),
                error=data.get("error"),
                source=data.get("source", "remote_worker"),
            )

        except requests.RequestException as exc:
            self.alive = False
            raise RuntimeError(f"Remote worker {self.id} failed: {exc}") from exc
        finally:
            self.active_connections -= 1

    def get_stats(self):
        requests = _requests()
        response = requests.get(
            f"{self.base_url}/stats",
            timeout=min(5.0, self.timeout_seconds),
        )
        response.raise_for_status()
        data = response.json()
        data["remote_url"] = self.base_url
        return data
