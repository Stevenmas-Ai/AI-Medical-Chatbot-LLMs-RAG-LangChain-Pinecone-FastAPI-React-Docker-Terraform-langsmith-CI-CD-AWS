"""
Monitoring & observability for the medical RAG pipeline.

Provides:
  * structured JSON logging of every request (latency, provider, token usage,
    retrieval scores, errors) — easy to ship to CloudWatch / Loki / Datadog.
  * an in-memory metrics registry exposed at /metrics in Prometheus text format.
  * a RAGTimer context manager + helpers to capture per-stage latency.
  * a feedback store for thumbs up/down so you can track answer quality in prod.

Designed to be zero-extra-infra by default. If LANGFUSE_PUBLIC_KEY (or
LANGCHAIN_API_KEY for LangSmith) is set, full traces can additionally be sent
there — see get_callbacks().
"""
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from threading import Lock

# --------------------------------------------------------------------------- #
# Structured logging
# --------------------------------------------------------------------------- #
logger = logging.getLogger("medical_rag")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def log_event(event: str, **fields):
    """Emit one structured JSON log line."""
    record = {"event": event, "ts": round(time.time(), 3), **fields}
    logger.info(json.dumps(record, default=str))


# --------------------------------------------------------------------------- #
# In-memory metrics registry (Prometheus text exposition format)
# --------------------------------------------------------------------------- #
class Metrics:
    def __init__(self):
        self._lock = Lock()
        self.counters = defaultdict(float)          # name|labels -> value
        self.histograms = defaultdict(list)         # name|labels -> [observations]

    def inc(self, name: str, value: float = 1.0, **labels):
        with self._lock:
            self.counters[self._key(name, labels)] += value

    def observe(self, name: str, value: float, **labels):
        with self._lock:
            self.histograms[self._key(name, labels)].append(value)

    @staticmethod
    def _key(name, labels):
        if not labels:
            return name
        lbl = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{lbl}}}"

    def render(self) -> str:
        """Render counters + histogram summaries in Prometheus text format."""
        lines = []
        with self._lock:
            for key, val in sorted(self.counters.items()):
                lines.append(f"{key} {val}")
            for key, obs in sorted(self.histograms.items()):
                if not obs:
                    continue
                n = len(obs)
                total = sum(obs)
                ordered = sorted(obs)
                p50 = ordered[int(0.50 * (n - 1))]
                p95 = ordered[int(0.95 * (n - 1))]
                base = key.split("{")[0]
                lines.append(f"{key.replace(base, base + '_count')} {n}")
                lines.append(f"{key.replace(base, base + '_sum')} {round(total, 6)}")
                lines.append(f"{key.replace(base, base + '_p50')} {round(p50, 6)}")
                lines.append(f"{key.replace(base, base + '_p95')} {round(p95, 6)}")
        return "\n".join(lines) + "\n"


metrics = Metrics()


# --------------------------------------------------------------------------- #
# Per-stage latency timer
# --------------------------------------------------------------------------- #
@contextmanager
def stage_timer(stage: str, **labels):
    """Time a pipeline stage, record it as a histogram, and return the elapsed."""
    start = time.perf_counter()
    holder = {"elapsed": 0.0}
    try:
        yield holder
    finally:
        elapsed = time.perf_counter() - start
        holder["elapsed"] = elapsed
        metrics.observe("rag_stage_latency_seconds", elapsed, stage=stage, **labels)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) when the provider doesn't return usage."""
    return max(1, len(text) // 4)


# Approximate USD pricing per 1K tokens (prompt, completion). Update as needed.
PRICING = {
    "groq":   (0.00005, 0.00008),   # llama-3.1-8b-instant
    "openai": (0.0025,  0.01),      # gpt-4o
}


def extract_token_usage(gen_response, question, context_docs, answer):
    """Prefer real usage_metadata from the LLM; fall back to a char-based estimate."""
    meta = getattr(gen_response, "usage_metadata", None) or {}
    if meta.get("input_tokens") and meta.get("output_tokens"):
        return {
            "prompt_tokens": meta["input_tokens"],
            "completion_tokens": meta["output_tokens"],
            "total_tokens": meta.get("total_tokens", meta["input_tokens"] + meta["output_tokens"]),
            "source": "provider",
        }
    ctx_text = " ".join(getattr(d, "page_content", "") for d in context_docs)
    pt = estimate_tokens(question + ctx_text)
    ct = estimate_tokens(answer)
    return {"prompt_tokens": pt, "completion_tokens": ct,
            "total_tokens": pt + ct, "source": "estimate"}


def record_cost(provider, usage):
    """Compute and record cost-per-query in USD from token usage."""
    p_rate, c_rate = PRICING.get(provider, (0.0, 0.0))
    cost = (usage["prompt_tokens"] / 1000 * p_rate
            + usage["completion_tokens"] / 1000 * c_rate)
    cost = round(cost, 6)
    metrics.inc("rag_tokens_total", usage["total_tokens"], provider=provider)
    metrics.inc("rag_cost_usd_total", cost, provider=provider)
    metrics.observe("rag_cost_usd_per_query", cost, provider=provider)
    return cost


def record_retrieval_scores(docs, provider="unknown"):
    """Pull relevance scores off retrieved docs, record them (for trend tracking),
    and return summary stats. Tracking each score as a histogram observation lets
    you alert when similarity scores trend down (retrieval degradation)."""
    scores = [
        d.metadata.get("relevance_score")
        for d in docs
        if getattr(d, "metadata", None) and d.metadata.get("relevance_score") is not None
    ]
    if not scores:
        return {}
    for s in scores:
        metrics.observe("rag_retrieval_relevance_score", float(s), provider=provider)
    return {
        "retrieval_score_max": round(max(scores), 4),
        "retrieval_score_min": round(min(scores), 4),
        "retrieval_score_mean": round(sum(scores) / len(scores), 4),
    }


# Backwards-compatible alias
def retrieval_score_stats(docs):
    return record_retrieval_scores(docs)


# --------------------------------------------------------------------------- #
# Feedback store (thumbs up/down). Swap for a DB in real prod.
# --------------------------------------------------------------------------- #
class FeedbackStore:
    def __init__(self):
        self._lock = Lock()
        self.up = 0
        self.down = 0
        self.records = []

    def record(self, rating: str, message: str = "", answer: str = ""):
        with self._lock:
            if rating == "up":
                self.up += 1
            elif rating == "down":
                self.down += 1
            self.records.append({"rating": rating, "message": message, "ts": time.time()})
        metrics.inc("rag_feedback_total", rating=rating)
        log_event("feedback", rating=rating, message=message)

    def summary(self):
        total = self.up + self.down
        return {
            "up": self.up,
            "down": self.down,
            "total": total,
            "satisfaction": round(self.up / total, 3) if total else None,
        }


feedback_store = FeedbackStore()


# --------------------------------------------------------------------------- #
# Optional hosted tracing (LangSmith / Langfuse) — only if env vars are set.
# --------------------------------------------------------------------------- #
def get_callbacks():
    """Return LangChain callbacks for hosted tracing if configured, else []."""
    callbacks = []
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse.callback import CallbackHandler
            callbacks.append(CallbackHandler())
        except Exception as e:  # pragma: no cover
            log_event("langfuse_init_failed", error=str(e))
    # LangSmith is enabled automatically when LANGCHAIN_TRACING_V2=true is set.
    return callbacks
