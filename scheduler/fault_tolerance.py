class FaultTolerance:
    def __init__(self, load_balancer):
        # Use the same load balancer as scheduler
        self.lb = load_balancer

    def handle_failure(self, request, retries=2):
        print(f"[FaultTolerance] Handling failure for request {request.id}")

        for attempt in range(retries):
            try:
                print(f"[FaultTolerance] Retry {attempt + 1} for request {request.id}")

                # Try dispatching again
                response = self.lb.dispatch(request)

                print(f"[FaultTolerance] Request {request.id} succeeded on retry {attempt + 1}")

                return response

            except Exception as e:
                print(f"[FaultTolerance] Retry {attempt + 1} failed: {e}")

        # If all retries fail
        print(f"[FaultTolerance] Request {request.id} FAILED permanently")

        return {
            "id": request.id,
            "result": "FAILED",
            "latency": None
        }