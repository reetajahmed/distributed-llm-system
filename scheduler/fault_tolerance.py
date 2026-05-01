from common.types import Response


def _response_succeeded(response):
    if isinstance(response, dict):
        return response.get("success", False)

    if hasattr(response, "success"):
        return bool(getattr(response, "success"))

    return getattr(response, "result", None) is not None


class FaultTolerance:
    def __init__(self, load_balancer):
        self.lb = load_balancer

    def handle_failure(self, request, retries=2):
        print(f"[FaultTolerance] Handling failure for request {request.id}")

        for attempt in range(retries):
            try:
                print(f"[FaultTolerance] Retry {attempt + 1} for request {request.id}")

                # IMPORTANT: pass request_id forward (already inside request)
                response = self.lb.dispatch(request)

                # ✅ Handle new format (dict or Response)
                success = _response_succeeded(response)

                if success:
                    print(f"[FaultTolerance] Request {request.id} succeeded on retry {attempt + 1}")
                    return response
                else:
                    print(f"[FaultTolerance] Retry {attempt + 1} returned unsuccessful result")

            except Exception as e:
                print(f"[FaultTolerance] Retry {attempt + 1} failed: {e}")

        print(f"[FaultTolerance] Request {request.id} FAILED permanently")

        # Return unified failure format
        return {
            "request_id": request.id,
            "answer": "FAILED",
            "success": False,
            "latency": 0.0,
            "rag_results": []
        }
