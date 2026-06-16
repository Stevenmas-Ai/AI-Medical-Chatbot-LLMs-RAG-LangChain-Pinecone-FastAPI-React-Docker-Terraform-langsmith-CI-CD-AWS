"""
Cross-encoder reranking for the medical RAG pipeline.

Bi-encoders (the all-MiniLM embedding model used for retrieval) encode the query
and each document *separately*, so they are fast but lose query-document
interaction. A cross-encoder encodes the (query, document) PAIR together and
outputs a single relevance score, which is far more accurate for ranking.

Strategy: retrieve a wide candidate set from Pinecone (e.g. k=10), then use the
cross-encoder to re-score and keep only the best top_n (e.g. 3) before sending
them to the LLM. This boosts context precision without changing the index.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2  (local, free, ~80MB CPU).
A Cohere rerank endpoint can be swapped in via build_cohere_reranker() if you
prefer a hosted reranker.
"""
import os

# Default local cross-encoder. Override with RERANKER_MODEL env var.
DEFAULT_RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def build_cross_encoder_reranker(top_n: int = 3, model_name: str = DEFAULT_RERANKER_MODEL):
    """Return a document compressor that reranks with a cross-encoder AND records
    the relevance score onto each returned doc's metadata['relevance_score'].

    LangChain's stock CrossEncoderReranker reorders but does NOT expose the score,
    so monitoring/observability can't track relevance trends. This subclass fixes
    that by attaching the score before returning the top_n docs.
    """
    from typing import Sequence
    from langchain_core.documents import Document
    from langchain_core.callbacks import Callbacks
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    class ScoringCrossEncoderReranker(CrossEncoderReranker):
        def compress_documents(self, documents: Sequence[Document], query: str,
                               callbacks: Callbacks = None) -> Sequence[Document]:
            scores = self.model.score([(query, d.page_content) for d in documents])
            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            out = []
            for doc, score in ranked[: self.top_n]:
                doc.metadata["relevance_score"] = float(score)
                out.append(doc)
            return out

    model = HuggingFaceCrossEncoder(model_name=model_name)
    return ScoringCrossEncoderReranker(model=model, top_n=top_n)


def build_reranking_retriever(base_retriever, top_n: int = 3,
                              model_name: str = DEFAULT_RERANKER_MODEL):
    """
    Wrap an existing retriever so results are reranked.

    Backend selection (env var RERANKER_BACKEND): "auto" (default), "cohere", "local".
      - "cohere"/"auto"+COHERE_API_KEY set -> hosted Cohere rerank (guide's pick)
      - otherwise -> local cross-encoder (free, no API key)

    base_retriever should fetch a WIDE candidate set (k=10-20) so the reranker
    has enough candidates to choose from.
    """
    from langchain.retrievers import ContextualCompressionRetriever

    backend = os.getenv("RERANKER_BACKEND", "auto").lower()
    use_cohere = backend == "cohere" or (backend == "auto" and os.getenv("COHERE_API_KEY"))

    if use_cohere:
        print("Reranker backend: Cohere (hosted)")
        compressor = build_cohere_reranker(top_n=top_n)
    else:
        print("Reranker backend: local cross-encoder")
        compressor = build_cross_encoder_reranker(top_n=top_n, model_name=model_name)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def build_cohere_reranker(top_n: int = 3, model: str = None):
    """Hosted Cohere reranker (the guide's recommendation).
    Requires COHERE_API_KEY and `pip install langchain-cohere`."""
    from langchain_cohere import CohereRerank
    model = model or os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
    return CohereRerank(model=model, top_n=top_n, cohere_api_key=os.getenv("COHERE_API_KEY"))


# ---------------------------------------------------------------------------
# Lightweight, dependency-minimal scorer used for demos / unit tests.
# Lets you rerank a plain list of (text) candidates without LangChain wiring.
# ---------------------------------------------------------------------------
def rerank_passages(query: str, passages, top_n: int = 3,
                    model_name: str = DEFAULT_RERANKER_MODEL):
    """Score (query, passage) pairs with the cross-encoder; return sorted list.

    Returns a list of dicts: {"rank", "score", "text"} ordered best-first.
    """
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)
    pairs = [(query, p) for p in passages]
    scores = model.predict(pairs)
    ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return [
        {"rank": i + 1, "score": float(s), "text": p}
        for i, (p, s) in enumerate(ranked[:top_n])
    ]
