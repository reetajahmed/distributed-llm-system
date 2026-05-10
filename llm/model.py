# Provides LLM inference functions for worker nodes

import time
import hashlib
import torch
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import threading

_llm_lock = threading.Lock()
_cache_lock = threading.RLock()
_generation_lock = threading.Lock()
_inflight = {}
_executor = ThreadPoolExecutor(max_workers=1)

from config import (
    LLM_MODEL_NAME,
    LLM_MAX_INPUT_TOKENS,
    LLM_MAX_NEW_TOKENS,
    LLM_TIMEOUT_SECONDS,
    LLM_CACHE_MAX_SIZE,
    LLM_ENABLE_LOGGING,
    LLM_MAX_RETRIES,
)
from common.text_utils import keyword_signature, normalize_text, short_hash


_tokenizer = None
_model = None

# Exact-answer and intent caches avoid repeated model inference.
_cache = OrderedDict()
_intent_cache = OrderedDict()
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _log(message: str):
    if LLM_ENABLE_LOGGING:
        print(f"[LLM] {message}")


def _load_llm():
    # Lazy-load the model once per process to avoid startup cost until needed.
    global _tokenizer, _model

    with _llm_lock:
        if _tokenizer is None or _model is None:
            _log(f"Loading model '{LLM_MODEL_NAME}' on {_device}...")
            _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
            _model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
            _model.to(_device)
            _model.eval()
            _log("Model loaded successfully.")

    return _tokenizer, _model


def _make_cache_key(query: str, context: str) -> str:
    # Exact cache key includes both the normalized query and retrieved context.
    raw = f"{normalize_text(query)}::{context.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _make_intent_key(query: str, context: str) -> str:
    # Intent key lets similar questions share an answer when context matches.
    signature = keyword_signature(query)
    context_hash = short_hash(context)
    return f"{signature}::{context_hash}"


def _split_intent_key(intent_key: str):
    signature, _, context_hash = intent_key.partition("::")
    return set(signature.split()), context_hash


def _similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _find_similar_intent_key(intent_key: str):
    # Search previous intent keys for a close-enough cache match.
    wanted_tokens, wanted_context_hash = _split_intent_key(intent_key)
    best_key = None
    best_score = 0.0

    for existing_key in _intent_cache:
        existing_tokens, existing_context_hash = _split_intent_key(existing_key)
        if existing_context_hash != wanted_context_hash:
            continue

        score = _similarity(wanted_tokens, existing_tokens)
        if score > best_score:
            best_key = existing_key
            best_score = score

    if best_score >= 0.34:
        return best_key

    return None


def _cache_get(query: str, context: str):
    # Try exact, same-intent, then similar-intent cache lookup.
    exact_key = _make_cache_key(query, context)
    intent_key = _make_intent_key(query, context)

    with _cache_lock:
        cache_key = exact_key
        if exact_key not in _cache and intent_key in _intent_cache:
            cache_key = _intent_cache[intent_key]
        elif exact_key not in _cache:
            similar_intent_key = _find_similar_intent_key(intent_key)
            if similar_intent_key is not None:
                cache_key = _intent_cache[similar_intent_key]

        if cache_key not in _cache:
            return None

        cached_result = _cache[cache_key].copy()
        _cache.move_to_end(cache_key)
        cached_result["cached"] = True
        cached_result["latency"] = 0
        return cached_result


def _add_to_cache(query: str, context: str, value: dict):
    # Store both exact and intent keys, evicting oldest entries when full.
    exact_key = _make_cache_key(query, context)
    intent_key = _make_intent_key(query, context)

    with _cache_lock:
        cached_value = value.copy()
        cached_value["cached"] = False
        cached_value["latency"] = 0
        _cache[exact_key] = cached_value
        _cache.move_to_end(exact_key)
        _intent_cache[intent_key] = exact_key
        _intent_cache.move_to_end(intent_key)

        while len(_cache) > LLM_CACHE_MAX_SIZE:
            _cache.popitem(last=False)
        while len(_intent_cache) > LLM_CACHE_MAX_SIZE:
            _intent_cache.popitem(last=False)


def _wait_for_inflight(cache_key: str):
    # Let one thread compute an answer while duplicate requests wait.
    with _cache_lock:
        event = _inflight.get(cache_key)
        if event is None:
            event = threading.Event()
            _inflight[cache_key] = event
            return event, True

    event.wait()
    return event, False


def _finish_inflight(cache_key: str):
    with _cache_lock:
        event = _inflight.pop(cache_key, None)
        if event is not None:
            event.set()


def _build_prompt(query: str, context: str = "") -> str:
    # RAG context is included only when retrieval found useful text.
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
    # Generation is locked because the local model is shared by all requests.
    tokenizer, model = _load_llm()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=LLM_MAX_INPUT_TOKENS,
    ).to(_device)

    input_tokens = int(inputs["input_ids"].shape[1])

    with _generation_lock:
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
            "source": "llm_error",
        }

    query = query.strip()
    context = context.strip() if context else ""

    cached_result = _cache_get(query, context)
    if cached_result is not None:
        latency = time.time() - start_time
        _log(f"Cache hit | request_id={request_id} | latency={latency:.4f}s")
        cached_result["request_id"] = request_id
        cached_result["latency"] = latency
        cached_result["source"] = "llm_cache"
        return cached_result

    inflight_key = _make_intent_key(query, context)
    _, owns_inflight = _wait_for_inflight(inflight_key)

    if not owns_inflight:
        cached_result = _cache_get(query, context)
        if cached_result is not None:
            latency = time.time() - start_time
            _log(
                f"Cache hit after wait | request_id={request_id} | "
                f"latency={latency:.4f}s"
            )
            cached_result["request_id"] = request_id
            cached_result["latency"] = latency
            cached_result["source"] = "llm_cache"
            return cached_result

    try:
        prompt = _build_prompt(query, context)

        # Retry protects callers from transient model or timeout failures.
        answer = None
        input_tokens = 0
        output_tokens = 0

        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                _log(
                    f"Running real model inference | "
                    f"attempt={attempt} | request_id={request_id}"
                )

                future = _executor.submit(_generate_answer, prompt)
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
        response = {
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
            "source": "llm_inference",
        }
        _add_to_cache(query, context, response)

        _log(
            f"Inference complete | request_id={request_id} | "
            f"latency={latency:.4f}s | "
            f"input_tokens={input_tokens} | output_tokens={output_tokens}"
        )

        return response

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
            "source": "llm_timeout",
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
            "source": "llm_error",
        }
    finally:
        if owns_inflight:
            _finish_inflight(inflight_key)


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
