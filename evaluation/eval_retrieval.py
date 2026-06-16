"""
Retrieval-quality evaluation WITHOUT needing an LLM API key.

Compares plain bi-encoder retrieval vs. cross-encoder reranking on the same
candidate set, reporting how the ranking changes. Useful as a fast, cheap CI
check that the reranker is wired up and improving ordering.

Reports, per query:
  * the bi-encoder top-3 (by embedding cosine similarity)
  * the cross-encoder top-3 (reranked from a wider candidate pool)
  * rank changes

Requires Pinecone + embeddings. For a fully offline demo with hand-written
passages, see demo_reranker.py.

Usage:  python -m evaluation.eval_retrieval
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
from src.reranker import build_cross_encoder_reranker

QUERIES = [
    "What are the common symptoms of asthma?",
    "What causes type 2 diabetes?",
    "What are the warning signs of a stroke?",
]


def main():
    embedding = download_hugging_face_embeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot", embedding=embedding
    )
    base = docsearch.as_retriever(search_kwargs={"k": 10})
    compressor = build_cross_encoder_reranker(top_n=3)

    for q in QUERIES:
        candidates = base.invoke(q)
        reranked = compressor.compress_documents(candidates, q)
        print("\n" + "=" * 80)
        print("QUERY:", q)
        print("-- bi-encoder top 3 (retrieval order) --")
        for i, d in enumerate(candidates[:3], 1):
            print(f"  {i}. {d.page_content[:90].strip()!r}")
        print("-- cross-encoder reranked top 3 --")
        for i, d in enumerate(reranked, 1):
            score = d.metadata.get("relevance_score")
            score_str = f"{score:.3f}" if score is not None else "n/a"
            print(f"  {i}. [score={score_str}] {d.page_content[:90].strip()!r}")


if __name__ == "__main__":
    main()
