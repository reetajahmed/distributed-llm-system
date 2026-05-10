from common.types import Response


def _response_succeeded(response):
    # Retry logic supports both dict and Response return styles.
    if isinstance(response, dict):
        return response.get("success", False)

    if hasattr(response, "success"):
        return bool(getattr(response, "success"))

    return getattr(response, "result", None) is not None


class FaultTolerance:
    def __init__(self, load_balancer):
        # Reuses the same load balancer so retries can choose another worker.
        self.lb = load_balancer

    def _attach_retry_metadata(self, response, attempts, recovered):
        # Add retry details without changing the response type.
        if isinstance(response, dict):
            response["retry_attempts"] = attempts
            response["recovered_by_retry"] = recovered
            response["source"] = response.get("source", "retry_failure")
            return response

        setattr(response, "retry_attempts", attempts)
        setattr(response, "recovered_by_retry", recovered)
        return response

    def handle_failure(self, request, retries=2):
        # Try dispatch again before giving up permanently.
        print(f"[FaultTolerance] Handling failure for request {request.id}")
        attempts = 0

        for attempt in range(retries):
            try:
                attempts = attempt + 1
                print(f"[FaultTolerance] Retry {attempt + 1} for request {request.id}")
                response = self.lb.dispatch(request)
                success = _response_succeeded(response)

                if success:
                    print(f"[FaultTolerance] Request {request.id} succeeded on retry {attempt + 1}")
                    return self._attach_retry_metadata(
                        response,
                        attempts=attempts,
                        recovered=True,
                    )
                else:
                    print(f"[FaultTolerance] Retry {attempt + 1} returned unsuccessful result")

            except Exception as e:
                attempts = attempt + 1
                print(f"[FaultTolerance] Retry {attempt + 1} failed: {e}")

        print(f"[FaultTolerance] Request {request.id} FAILED permanently")

        return {
            "request_id": request.id,
            "answer": "FAILED",
            "success": False,
            "latency": 0.0,
            "rag_results": [],
            "retry_attempts": attempts,
            "recovered_by_retry": False,
            "source": "retry_failure",
        }
