import time
from scheduler.fault_tolerance import FaultTolerance


def _response_succeeded(response):
    if isinstance(response, dict):
        return response.get("success", False)

    if hasattr(response, "success"):
        return bool(getattr(response, "success"))

    return getattr(response, "result", None) is not None


def _response_strategy(response):
    if isinstance(response, dict):
        return response.get("strategy_used", "unknown")

    return getattr(response, "strategy_used", "unknown")


class Scheduler:
    def __init__(self, load_balancer):
        self.lb = load_balancer
        self.fault_handler = FaultTolerance(self.lb)

        # Tracking
        self.active_tasks = {}   # {request_id: status}
        self.results = {}        # {request_id: response}

        # Stats (for dashboard)
        self.completed = 0
        self.failed = 0
        self.retried = 0

    def handle_request(self, request):
        print(f"[Scheduler] Received request {request.id}")

        # 🟡 Add better status system
        self.active_tasks[request.id] = "RUNNING"

        start_time = time.time()

        try:
            response = self.lb.dispatch(request)

            # ✅ Handle NEW response format
            success = _response_succeeded(response)
            strategy = _response_strategy(response)

            print(f"[Scheduler] Strategy used: {strategy}")

            if success:
                self.active_tasks[request.id] = "COMPLETED"
                self.completed += 1
            else:
                raise Exception("Worker returned unsuccessful result")

            self.results[request.id] = response

            latency = time.time() - start_time
            print(f"[Scheduler] Request {request.id} completed in {latency:.3f}s")

            return response

        except Exception as e:
            print(f"[Scheduler] ERROR on request {request.id}: {e}")

            # 🔴 Mark as retrying
            self.active_tasks[request.id] = "RETRYING"
            self.retried += 1

            # Call fault tolerance
            response = self.fault_handler.handle_failure(request)

            # Check retry result
            success = _response_succeeded(response)

            if success:
                self.active_tasks[request.id] = "REASSIGNED"
                self.completed += 1
                self.results[request.id] = response

                print(f"[Recovery] Request {request.id} reassigned successfully")
            else:
                self.active_tasks[request.id] = "FAILED"
                self.failed += 1

            return response

    def print_status(self):
        print("\n===== Scheduler Status =====")
        for req_id, status in self.active_tasks.items():
            print(f"Request {req_id}: {status}")
        print("============================\n")

    # 🔥 Dashboard (FULL MARK feature)
    def print_summary(self):
        print("\n===== Scheduler Summary =====")
        print(f"Tasks completed: {self.completed}")
        print(f"Tasks retried: {self.retried}")
        print(f"Tasks failed: {self.failed}")
        print("============================\n")
