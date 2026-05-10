import argparse
from dataclasses import asdict
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from common.default_docs import DEFAULT_DOCUMENTS
from common.types import Request
from workers.gpu_worker import GPUWorker
from workers.rag import ingest_documents


class ProcessRequest(BaseModel):
    # JSON body accepted by the /process endpoint.
    id: int
    query: str
    timestamp: Optional[float] = None


def create_app(worker_id: int, ingest_rag: bool = True) -> FastAPI:
    # Each HTTP server wraps one GPUWorker instance.
    app = FastAPI(title=f"GPU Worker {worker_id}")
    worker = GPUWorker(worker_id=worker_id)

    if ingest_rag:
        ingest_result = ingest_documents(DEFAULT_DOCUMENTS)
        print(
            f"[WorkerServer {worker_id}] RAG documents: "
            f"{ingest_result['total_documents']} total "
            f"({ingest_result['added']} added)"
        )

    @app.get("/")
    def root():
        return {
            "service": "distributed-llm-worker",
            "worker_id": worker.id,
            "alive": worker.alive,
            "endpoints": {
                "health": "/health",
                "stats": "/stats",
                "process": "/process",
                "docs": "/docs",
            },
        }

    @app.get("/health")
    def health():
        return {
            "worker_id": worker.id,
            "alive": worker.alive,
            "active_connections": worker.active_connections,
        }

    @app.get("/stats")
    def stats():
        return worker.get_stats()

    @app.post("/process")
    def process(payload: ProcessRequest):
        # Rebuild the shared Request dataclass before calling worker.process.
        request = Request(id=payload.id, query=payload.query)
        if payload.timestamp is not None:
            request.timestamp = payload.timestamp

        response = worker.process(request)
        return asdict(response)

    return app


def parse_args():
    # Worker id and port are required so multiple servers can run locally.
    parser = argparse.ArgumentParser(description="Run one distributed GPU worker.")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--skip-rag-ingest",
        action="store_true",
        help="Do not seed this worker's local RAG index.",
    )
    return parser.parse_args()


def main():
    # Start one FastAPI/uvicorn worker service.
    args = parse_args()
    app = create_app(
        worker_id=args.worker_id,
        ingest_rag=not args.skip_rag_ingest,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
