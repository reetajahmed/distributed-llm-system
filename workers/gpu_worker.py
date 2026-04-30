import time
from llm.model import run_llm
from workers.rag import retrieve_context
from common.types import Response


class GPUWorker:
    def __init__(self, worker_id: int):
        self.id = worker_id
        self.alive = True                 
        self.active_connections = 0       

    def process(self, request):
        start = time.time()
        self.active_connections += 1     

        print(f"[Worker {self.id}] Processing request {request.id}")

        try:
            context = retrieve_context(request.query)
            result = run_llm(request.query, context)

        except Exception as e:
            print(f"[Worker {self.id}] ERROR: {e}")
            result = f"Worker failed safely: {str(e)}"

        finally:
            self.active_connections -= 1  

        latency = time.time() - start

        response = Response(
            id=request.id,
            worker_id=self.id,
            result=result,
            latency=latency
        )

        print(f"[Worker {self.id}] Finished request {request.id} in {latency:.3f}s")

        return response