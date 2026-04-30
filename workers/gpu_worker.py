import time
from llm.model import run_llm          # Member 4’s AI module
from workers.rag import retrieve_context  # Your RAG pipeline

class GPUWorker:
    def __init__(self, worker_id: int):
        self.id = worker_id

    def process(self, request):
        start = time.time()
        print(f"[Worker {self.id}] Processing request {request.id}")

        # Step 1: Retrieve semantic context (RAG pipeline)
        context = retrieve_context(request.query)

        # Step 2: Run LLM inference with retrieved context
        result = run_llm(request.query, context)

        latency = time.time() - start

        response = {
            "id": request.id,
            "worker_id": self.id,
            "result": result,
            "latency": latency
        }

        print(f"[Worker {self.id}] Finished request {request.id} in {latency:.3f}s")
        return response