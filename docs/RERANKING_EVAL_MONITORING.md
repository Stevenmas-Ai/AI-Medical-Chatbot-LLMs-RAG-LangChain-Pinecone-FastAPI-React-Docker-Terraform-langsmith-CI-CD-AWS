# Reranking, Evaluation & Observability

Three production-grade additions to the medical RAG pipeline.

## 1. Cross-Encoder Reranking

**Why:** The retriever uses a bi-encoder (`all-MiniLM-L6-v2`) that encodes the
query and documents separately — fast, but it misses query–document interaction.
A cross-encoder scores the `(query, document)` pair *together*, so it ranks far
more accurately.

**How:** `app.py` now retrieves a **wide** candidate set (`k=10`) from Pinecone,
then a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, local & free)
reranks down to the **top 3** that go to the LLM.

- Code: [`src/reranker.py`](../src/reranker.py)
- Toggle: `USE_RERANKER=false` falls back to plain `k=3` similarity (for A/B tests).
- Hosted alternative: `build_cohere_reranker()` (needs `COHERE_API_KEY`).

**Sample output** (`python -m evaluation.demo_reranker`):

```
QUERY: What are the common symptoms of asthma?

Cross-encoder reranked TOP 3:
  rank 1  score=7.899  Asthma symptoms include wheezing, shortness of breath, chest tightness...
  rank 2  score=2.910  During an asthma attack the airways narrow and swell and produce extra mucus...
  rank 3  score=-0.586 Asthma is a chronic disease of the airways and affects millions of people...
```

The two passages that actually describe *symptoms* are surfaced; the off-topic
treatment / diabetes / exercise candidates are dropped.

## 2. Evaluation (LangSmith)

Uses the official LangSmith RAG-eval pattern
(https://docs.langchain.com/langsmith/evaluate-rag-tutorial), adapted to this
stack (Groq judge + Pinecone + reranker). Four LLM-as-judge evaluators score
each answer pass/fail, aggregated as **% pass**:

| Evaluator | Question it answers |
|-----------|---------------------|
| **correctness** | Is the answer factually correct vs. the reference? |
| **groundedness** | Is the answer grounded in the retrieved docs (no hallucination)? |
| **relevance** | Does the answer address the question? |
| **retrieval_relevance** | Are the retrieved docs relevant to the question? |

- Code: [`evaluation/langsmith_eval.py`](../evaluation/langsmith_eval.py)
- Dataset: [`evaluation/eval_dataset.json`](../evaluation/eval_dataset.json) — 8 medical
  Q/A pairs (expand to 50–100 for real coverage); auto-uploaded to LangSmith.
- [`evaluation/eval_retrieval.py`](../evaluation/eval_retrieval.py) /
  [`demo_reranker.py`](../evaluation/demo_reranker.py) — quick reranker checks, no API key.

Requires `LANGSMITH_API_KEY` (free from smith.langchain.com) + `GROQ_API_KEY` + `PINECONE_API_KEY`.

```bash
python -m evaluation.langsmith_eval                  # reranker ON
USE_RERANKER=false python -m evaluation.langsmith_eval    # baseline, then compare
```

Results (and full pipeline traces) appear in the LangSmith dashboard. Sample:

```
correctness          100.0%  (3/3)
groundedness         100.0%  (3/3)
relevance            100.0%  (3/3)
retrieval_relevance  100.0%  (3/3)
```

## 3. Monitoring & Observability

Code: [`src/monitoring.py`](../src/monitoring.py). Zero extra infrastructure by default.

The `/chat` handler splits the pipeline into two separately-timed stages so you
can see exactly where time goes. The five metrics the guide calls for:

| Metric (guide) | How it's tracked |
|----------------|------------------|
| **Retrieval latency** | `rag_stage_latency_seconds{stage="retrieval"}` — vector search + rerank, timed alone |
| **Generation latency** | `rag_stage_latency_seconds{stage="generation"}` — LLM response, timed alone |
| **Retrieval relevance scores** | `rag_retrieval_relevance_score` histogram (every score recorded → alert if p50 trends down) |
| **User feedback** | `rag_feedback_total{rating="up"|"down"}` via `POST /feedback` |
| **Token usage / cost** | `rag_tokens_total`, `rag_cost_usd_total`, `rag_cost_usd_per_query` (real `usage_metadata`, falls back to estimate) |

| Endpoint | Purpose |
|----------|---------|
| `GET /metrics` | Prometheus-format counters + latency/score/cost histograms (p50/p95) |
| `GET /health` | liveness + reranker flag + feedback summary |
| `POST /feedback` | record thumbs up/down per answer |

Set `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` or `LANGCHAIN_TRACING_V2=true`
to additionally stream full traces to Langfuse / LangSmith (`get_callbacks()`
is already passed into both pipeline stages).

**Sample `/metrics` output (real, from the instrumentation):**

```
rag_requests_total{provider="groq"} 2.0
rag_tokens_total{provider="openai"} 372.0
rag_cost_usd_total{provider="openai"} 0.00138
rag_feedback_total{rating="up"} 2.0
rag_feedback_total{rating="down"} 1.0
rag_stage_latency_seconds_p50{provider="groq",stage="retrieval"} 0.018
rag_stage_latency_seconds_p50{provider="groq",stage="generation"} 0.4
rag_stage_latency_seconds_p95{provider="openai",stage="total"} 1.12
rag_retrieval_relevance_score_p50{provider="groq"} 2.9
rag_retrieval_relevance_score_p95{provider="groq"} 6.1
```

**Sample structured log line (per `/chat`):**

```json
{"event": "chat_completed", "provider": "openai", "retrieval_latency_s": 0.02,
 "generation_latency_s": 1.1, "total_latency_s": 1.12, "prompt_tokens": 312,
 "completion_tokens": 60, "cost_usd": 0.00138, "retrieval_score_max": 8.4,
 "retrieval_score_mean": 4.8667}
```
