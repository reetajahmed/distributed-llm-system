# Provides run_llm(query, context) for worker nodes
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-small"

_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_llm():
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        print(f"[LLM] Loading model on {_device}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()
        print("[LLM] Model loaded successfully.")

    return _tokenizer, _model

def run_llm(query: str, context: str = "") -> str:
    """
    Run real HuggingFace LLM inference.

    The model is loaded once and reused for all requests.
    This is important for performance under concurrent workloads.
    """
    if not query or not query.strip():
        return "No query provided."

    try:
        tokenizer, model = _load_llm()

        prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{query}

Answer:
"""

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

    except Exception as e:
        return f"LLM inference failed safely: {str(e)}"