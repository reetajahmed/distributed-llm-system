# distributed-llm-system

## AI Layer
The AI layer provides two functions used by GPU workers:

```python
from workers.rag import retrieve_context
from llm.model import run_llm

context = retrieve_context(request.query)
answer = run_llm(request.query, context)
```

Function Contracts
    -retrieve_context(query: str, top_k: int = 3) -> str
        Retrieves the most relevant context from the project knowledge base.
    -run_llm(query: str, context: str = "") -> str
        Runs real HuggingFace LLM inference and returns the generated answer.

Performance Note
    The RAG embedding model and HuggingFace LLM are loaded once and reused across requests.
    Workers must not reload the model per request because that would cause high latency and memory usage.

Fault Safety
    run_llm() includes safe error handling. If inference fails, it returns a controlled error message instead of crashing the worker.

## Load Tests

Run a local in-process load test:

```bash
python main.py
```

The default workload generates many unique realistic queries and some repeats:

```bash
python main.py --query-unique-ratio 0.75 --query-repeat-pool-size 50
```

For a heavier worker/LLM stress test:

```bash
python main.py --query-unique-ratio 1.0
```

For a cache-heavy comparison:

```bash
python main.py --query-unique-ratio 0.2 --query-repeat-pool-size 20
```

## Distributed Worker Mode

Start three worker processes in separate terminals. Each command keeps running:

```bash
python worker_server.py --worker-id 1 --port 8001
python worker_server.py --worker-id 2 --port 8002
python worker_server.py --worker-id 3 --port 8003
```

Health and stats endpoints:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/stats
```

Run the scheduler/load test against the worker services:

```bash
python main.py --distributed --worker-urls http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003
```

Short distributed smoke test:

```bash
python main.py --distributed --users 100 --query-workload-size 100 --query-unique-ratio 0.5
```

The final report includes scheduler source counts, retry attempts/successes/failures, logical routed failures, exception routed failures, and per-worker `/stats`.

## CSV Reports

Each `main.py` run exports documentation-friendly CSV tables under `reports/<timestamp>/` by default:

```text
run_summary.csv
request_results.csv
load_balancer_metrics.csv
load_balancer_source_counts.csv
requests_per_worker.csv
scheduler_source_counts.csv
worker_stats.csv
```

Use a different export directory:

```bash
python main.py --distributed --export-dir experiment_reports
```

Disable CSV export:

```bash
python main.py --distributed --export-dir ""
```
