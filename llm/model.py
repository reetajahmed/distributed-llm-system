# Provides LLM inference functions for worker nodes

import time
import hashlib
import torch
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import (
    LLM_MODEL_NAME,
    LLM_MAX_INPUT_TOKENS,
    LLM_MAX_NEW_TOKENS,
    LLM_TIMEOUT_SECONDS,
    LLM_CACHE_MAX_SIZE,
    LLM_ENABLE_LOGGING,
    LLM_MAX_RETRIES,
)


_tokenizer = None
_model = None
_cache = OrderedDict()
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _log(message: str):
    if LLM_ENABLE_LOGGING:
        print(f"[LLM] {message}")


def _load_llm():
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        _log(f"Loading model '{LLM_MODEL_NAME}' on {_device}...")
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
        _model.to(_device)
        _model.eval()
        _log("Model loaded successfully.")

    return _tokenizer, _model


def _make_cache_key(query: str, context: str) -> str:
    raw = f"{query.strip()}::{context.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _add_to_cache(key: str, value: str):
    _cache[key] = value
    _cache.move_to_end(key)

    if len(_cache) > LLM_CACHE_MAX_SIZE:
        _cache.popitem(last=False)


def _build_prompt(query: str, context: str = "") -> str:
    if context and context.strip():
        return f"""
Answer the question using only the context below.

Context:
{context}

Question:
{query}

Answer:
"""
    return f"""
Answer the following question clearly.

Question:
{query}

Answer:
"""


def _generate_answer(prompt: str) -> tuple[str, int, int]:
    tokenizer, model = _load_llm()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=LLM_MAX_INPUT_TOKENS,
    ).to(_device)

    input_tokens = int(inputs["input_ids"].shape[1])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    output_tokens = int(outputs.shape[1])

    return answer, input_tokens, output_tokens


def run_llm(query: str, context: str = "", request_id: str = None) -> str:
    """
    Backward-compatible function for existing worker code.
    Returns only the answer string.
    """
    result = run_llm_with_metrics(query, context, request_id)
    return result["answer"]


def run_llm_with_metrics(
    query: str,
    context: str = "",
    request_id: str = None,
) -> dict:
    """
    Runs real HuggingFace LLM inference with:
    - caching
    - timeout protection
    - retry logic
    - request tracing
    - metrics
    - safe error handling
    """
    start_time = time.time()

    if not query or not query.strip():
        return {
            "request_id": request_id,
            "answer": "No query provided.",
            "success": False,
            "cached": False,
            "latency": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "device": _device,
            "model": LLM_MODEL_NAME,
            "error": "empty_query",
        }

    query = query.strip()
    context = context.strip() if context else ""

    cache_key = _make_cache_key(query, context)

    if cache_key in _cache:
        cached_answer = _cache[cache_key]
        _cache.move_to_end(cache_key)

        latency = time.time() - start_time
        _log(f"Cache hit | request_id={request_id} | latency={latency:.4f}s")

        return {
            "request_id": request_id,
            "answer": cached_answer,
            "success": True,
            "cached": True,
            "latency": latency,
            "input_tokens": 0,
            "output_tokens": len(cached_answer.split()),
            "device": _device,
            "model": LLM_MODEL_NAME,
            "error": None,
        }

    try:
        prompt = _build_prompt(query, context)

        answer = None
        input_tokens = 0
        output_tokens = 0

        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                _log(
                    f"Running real model inference | "
                    f"attempt={attempt} | request_id={request_id}"
                )

                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_generate_answer, prompt)
                    answer, input_tokens, output_tokens = future.result(
                        timeout=LLM_TIMEOUT_SECONDS
                    )

                break

            except TimeoutError as e:
                _log(
                    f"Inference timeout | attempt={attempt} | "
                    f"request_id={request_id}"
                )

                if attempt == LLM_MAX_RETRIES:
                    raise e

            except Exception as e:
                _log(
                    f"Inference error | attempt={attempt} | "
                    f"request_id={request_id} | error={str(e)}"
                )

                if attempt == LLM_MAX_RETRIES:
                    raise e

        latency = time.time() - start_time
        _add_to_cache(cache_key, answer)

        _log(
            f"Inference complete | request_id={request_id} | "
            f"latency={latency:.4f}s | "
            f"input_tokens={input_tokens} | output_tokens={output_tokens}"
        )

        return {
            "request_id": request_id,
            "answer": answer,
            "success": True,
            "cached": False,
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "device": _device,
            "model": LLM_MODEL_NAME,
            "error": None,
        }

    except TimeoutError:
        latency = time.time() - start_time
        _log(
            f"Inference failed after retries due to timeout | "
            f"request_id={request_id}"
        )

        return {
            "request_id": request_id,
            "answer": "LLM inference timed out safely.",
            "success": False,
            "cached": False,
            "latency": latency,
            "input_tokens": 0,
            "output_tokens": 0,
            "device": _device,
            "model": LLM_MODEL_NAME,
            "error": "timeout",
        }

    except Exception as e:
        latency = time.time() - start_time
        _log(
            f"Inference failed safely | request_id={request_id} | "
            f"error={str(e)}"
        )

        return {
            "request_id": request_id,
            "answer": f"LLM inference failed safely: {str(e)}",
            "success": False,
            "cached": False,
            "latency": latency,
            "input_tokens": 0,
            "output_tokens": 0,
            "device": _device,
            "model": LLM_MODEL_NAME,
            "error": str(e),
        }


def run_llm_batch(requests: list[dict]) -> list[dict]:
    """
    Batch-ready interface.

    Current implementation processes requests sequentially,
    but the interface allows future true batching without changing worker code.

    Expected input:
    [
        {
            "request_id": "req_1",
            "query": "...",
            "context": "..."
        }
    ]
    """
    results = []

    for request in requests:
        request_id = request.get("request_id")
        query = request.get("query", "")
        context = request.get("context", "")

        result = run_llm_with_metrics(
            query=query,
            context=context,
            request_id=request_id,
        )

        results.append(result)

    return results