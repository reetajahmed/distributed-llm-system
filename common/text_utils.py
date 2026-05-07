import hashlib
import re


_STOPWORDS = {
    "a",
    "an",
    "and",
    "about",
    "are",
    "can",
    "define",
    "describe",
    "does",
    "explain",
    "for",
    "give",
    "how",
    "in",
    "is",
    "me",
    "of",
    "please",
    "tell",
    "the",
    "to",
    "what",
    "whats",
    "work",
    "working",
}

_SYNONYMS = {
    "ai": "artificial-intelligence",
    "artificial": "artificial-intelligence",
    "augmented": "rag",
    "balancer": "balance",
    "balancing": "balance",
    "compute": "computing",
    "distributed": "distributed",
    "distribution": "distributed",
    "failure": "fault",
    "failures": "fault",
    "fault": "fault",
    "generation": "rag",
    "llm": "llm",
    "intelligence": "artificial-intelligence",
    "load": "load",
    "rag": "rag",
    "recovery": "fault",
    "recover": "fault",
    "resilience": "fault",
    "resilient": "fault",
    "retrieval": "rag",
    "retrieve": "rag",
    "server": "worker",
    "servers": "worker",
    "tolerance": "tolerance",
    "tolerant": "tolerance",
    "workers": "worker",
}


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def keyword_signature(text: str) -> str:
    tokens = []
    for token in normalize_text(text).split():
        if token in _STOPWORDS:
            continue
        if token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]
        tokens.append(_SYNONYMS.get(token, token))

    return " ".join(sorted(set(tokens)))


def short_hash(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()[:16]
