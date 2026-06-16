"""
LangSmith RAG evaluation — official pattern from
https://docs.langchain.com/langsmith/evaluate-rag-tutorial
adapted to THIS chatbot's stack (Groq LLMs + Pinecone + cross-encoder/Cohere rerank).

Differences from the tutorial (all because our stack isn't OpenAI/in-memory):
  * Answer model  -> ChatGroq llama-3.1-8b-instant (the chatbot's real model)
  * Judge model   -> ChatGroq llama-3.3-70b-versatile (stronger grader)
  * Retriever     -> our Pinecone + cross-encoder/Cohere reranking retriever
  * Dataset       -> built from evaluation/eval_dataset.json (question + ground_truth)
  * Structured output uses Groq's function-calling (json_schema/strict is OpenAI-only)

Four LLM-as-judge evaluators (boolean pass/fail -> aggregate = % pass):
  correctness          : answer vs. ground-truth reference
  relevance            : answer vs. question
  groundedness         : answer vs. retrieved docs (hallucination check)
  retrieval_relevance  : retrieved docs vs. question

Setup:
    pip install langsmith
    Add to .env:  LANGSMITH_API_KEY=ls__...   (free key from smith.langchain.com)
    (GROQ_API_KEY + PINECONE_API_KEY already required)

Usage:
    python -m evaluation.langsmith_eval
    USE_RERANKER=false python -m evaluation.langsmith_eval     # baseline
    RERANKER_BACKEND=cohere python -m evaluation.langsmith_eval # Cohere rerank
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Turn on LangSmith tracing so the runs + traces show up in the dashboard.
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "medical-chatbot-rag")

from langsmith import Client, traceable
from langchain_groq import ChatGroq

from src.helper import download_hugging_face_embeddings
from src.reranker import build_reranking_retriever

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
DATASET_NAME = os.getenv("LANGSMITH_DATASET", "medical-chatbot-qa")
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
TOP_N = int(os.getenv("TOP_N", "3"))
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))

client = Client()


# --------------------------------------------------------------------------- #
# Retriever (Pinecone + optional rerank), built once.
# --------------------------------------------------------------------------- #
def build_retriever():
    from langchain_pinecone import PineconeVectorStore
    embedding = download_hugging_face_embeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot", embedding=embedding
    )
    if USE_RERANKER:
        base = docsearch.as_retriever(search_kwargs={"k": CANDIDATE_K})
        return build_reranking_retriever(base, top_n=TOP_N)
    return docsearch.as_retriever(search_kwargs={"k": TOP_N})


retriever = build_retriever()


# --------------------------------------------------------------------------- #
# Target: the RAG bot under test. @traceable streams the trace to LangSmith.
# --------------------------------------------------------------------------- #
answer_llm = ChatGroq(model="llama-3.1-8b-instant",
                      api_key=os.getenv("GROQ_API_KEY"), temperature=0)


@traceable()
def rag_bot(question: str) -> dict:
    docs = retriever.invoke(question)
    docs_string = "\n\n".join(doc.page_content for doc in docs)
    instructions = (
        "You are a Medical assistant. Answer the question using ONLY the source "
        "documents below. Do not use outside knowledge. If the answer is not in "
        "the documents, say you don't know. Use three sentences maximum.\n\n"
        f"<context>\n{docs_string}\n</context>"
    )
    ai_msg = answer_llm.invoke([
        {"role": "system", "content": instructions},
        {"role": "user", "content": question},
    ])
    return {"answer": ai_msg.content, "documents": docs}


def target(inputs: dict) -> dict:
    return rag_bot(inputs["question"])


# --------------------------------------------------------------------------- #
# LLM-as-judge grader (Groq 70b). Groq uses function-calling for structured
# output (OpenAI's json_schema/strict is not supported), so we omit `method`.
# --------------------------------------------------------------------------- #
# Judge model is configurable. 70b is the most accurate grader, but it has a
# small free-tier daily token budget; 8b-instant has a much larger one.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.1-8b-instant")

_judge_llm = ChatGroq(model=JUDGE_MODEL, api_key=os.getenv("GROQ_API_KEY"), temperature=0)


def _grade_bool(instructions: str, user_msg: str, field: str) -> bool:
    """Model-agnostic structured grading.

    Groq's tool-calling is unreliable on smaller models (it emits
    `<function=...>{...}</function>` text that fails strict validation), so instead
    of with_structured_output we prompt for raw JSON and parse it robustly,
    recovering the boolean even from the wrapped/fenced variants.
    """
    import json as _json
    import re
    sys_prompt = (
        instructions
        + f'\n\nRespond with ONLY a JSON object, no other text, in exactly this form:\n'
          f'{{"{field}": true or false, "explanation": "<your step-by-step reasoning>"}}'
    )
    resp = _judge_llm.invoke([
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ])
    text = resp.content if hasattr(resp, "content") else str(resp)

    # Try strict JSON first, then any {...} blob, then a bare `"field": true/false`.
    for candidate in ([text] + re.findall(r"\{.*?\}", text, re.DOTALL)):
        try:
            data = _json.loads(candidate)
            if field in data:
                return bool(data[field])
        except Exception:
            pass
    m = re.search(rf'"{field}"\s*:\s*(true|false)', text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    raise ValueError(f"Could not parse '{field}' from judge output: {text[:200]}")


# ---- 1. Correctness: answer vs. reference ----
correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer.
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Correctness:
True means that the student's answer meets all of the criteria.
False means that the student's answer does not meet all of the criteria.

Explain your reasoning step-by-step. Avoid simply stating the correct answer at the outset."""


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """RAG answer accuracy vs. the reference answer."""
    msg = (f"QUESTION: {inputs['question']}\n"
           f"GROUND TRUTH ANSWER: {reference_outputs['answer']}\n"
           f"STUDENT ANSWER: {outputs['answer']}")
    return _grade_bool(correctness_instructions, msg, "correct")


# ---- 2. Relevance: answer vs. question ----
relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
True means that the student's answer meets all of the criteria.
False means that the student's answer does not meet all of the criteria.

Explain your reasoning step-by-step. Avoid simply stating the correct answer at the outset."""


def relevance(inputs: dict, outputs: dict) -> bool:
    """RAG answer helpfulness/relevance to the question."""
    msg = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    return _grade_bool(relevance_instructions, msg, "relevant")


# ---- 3. Groundedness: answer vs. retrieved docs (hallucination) ----
grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS.
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
True means that the student's answer meets all of the criteria.
False means that the student's answer does not meet all of the criteria.

Explain your reasoning step-by-step. Avoid simply stating the correct answer at the outset."""


def groundedness(inputs: dict, outputs: dict) -> bool:
    """Is the answer grounded in the retrieved documents (no hallucination)."""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    msg = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    return _grade_bool(grounded_instructions, msg, "grounded")


# ---- 4. Retrieval relevance: retrieved docs vs. question ----
retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) Your goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning step-by-step. Avoid simply stating the correct answer at the outset."""


def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """Are the retrieved documents relevant to the question."""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    msg = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
    return _grade_bool(retrieval_relevance_instructions, msg, "relevant")


# --------------------------------------------------------------------------- #
# Dataset: create in LangSmith from local JSON (idempotent).
# --------------------------------------------------------------------------- #
def ensure_dataset() -> str:
    samples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    limit = int(os.getenv("EVAL_LIMIT", "0"))
    if limit:
        samples = samples[:limit]

    if client.has_dataset(dataset_name=DATASET_NAME):
        return DATASET_NAME

    ds = client.create_dataset(dataset_name=DATASET_NAME,
                               description="Medical chatbot RAG QA eval set")
    client.create_examples(
        dataset_id=ds.id,
        inputs=[{"question": s["question"]} for s in samples],
        outputs=[{"answer": s["ground_truth"]} for s in samples],
    )
    print(f"Created LangSmith dataset '{DATASET_NAME}' with {len(samples)} examples.")
    return DATASET_NAME


def main():
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("Set LANGSMITH_API_KEY in .env (free key from smith.langchain.com).")

    dataset_name = ensure_dataset()
    print(f"Evaluating (reranker={USE_RERANKER}, backend={os.getenv('RERANKER_BACKEND','auto')})...")

    results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=[correctness, groundedness, relevance, retrieval_relevance],
        experiment_prefix=f"medical-rag-rerank-{USE_RERANKER}",
        max_concurrency=1,   # serialize -> respect Groq free-tier rate limits
        metadata={"reranker": USE_RERANKER,
                  "backend": os.getenv("RERANKER_BACKEND", "auto"),
                  "answer_model": "llama-3.1-8b-instant",
                  "judge_model": "llama-3.3-70b-versatile"},
    )

    # Print aggregate pass-rates from the results dataframe.
    try:
        import pandas as pd
        df = results.to_pandas()
        metric_names = ["correctness", "groundedness", "relevance", "retrieval_relevance"]
        print("\n=== LANGSMITH RESULTS (% pass) ===")
        for metric in metric_names:
            # column may be "feedback.<metric>" or "<metric>" depending on version
            col = next((c for c in df.columns
                        if c == metric or c.endswith("." + metric)), None)
            if col is None:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) == 0:
                print(f"  {metric:<22} n/a (no scores)")
            else:
                print(f"  {metric:<22} {vals.mean() * 100:.1f}%  ({int(vals.sum())}/{len(vals)})")
        print("\nFull traces + comparison in the LangSmith UI: https://smith.langchain.com")
    except Exception as e:
        print(f"(Couldn't render local summary: {e}) — see results in the LangSmith UI.")
    return results


if __name__ == "__main__":
    main()
