import time
from scheduler.fault_tolerance import FaultTolerance


class Scheduler:
    def __init__(self, load_balancer):
        self.lb = load_balancer

        self.fault_handler = FaultTolerance(self.lb)

        self.active_tasks = {}        # {request_id: status}
        self.results = {}             # {request_id: response}

    def handle_request(self, request):
        print(f"[Scheduler] Received request {request.id}")

        # Mark as processing
        self.active_tasks[request.id] = "PROCESSING"

        start_time = time.time()

        try:
            # Send to load balancer → worker
            response = self.lb.dispatch(request)

            # Mark as done
            self.active_tasks[request.id] = "DONE"

            # Store result
            self.results[request.id] = response

            print(f"[Scheduler] Strategy used: {response.strategy_used}")

            latency = time.time() - start_time
            print(f"[Scheduler] Request {request.id} completed in {latency:.3f}s")

            return response

        except Exception as e:
            print(f"[Scheduler] ERROR on request {request.id}: {e}")

            # Mark as failed
            self.active_tasks[request.id] = "FAILED"

            # Call fault tolerance handler
            response = self.fault_handler.handle_failure(request)

            # Update result after retry
            if response and response.result != "FAILED":
                self.active_tasks[request.id] = "DONE"
                self.results[request.id] = response
            else:
                self.active_tasks[request.id] = "FAILED"

            return response

    def print_status(self):
        print("\n===== Scheduler Status =====")
        for req_id, status in self.active_tasks.items():
            print(f"Request {req_id}: {status}")
        print("============================\n")