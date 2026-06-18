import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt
from src.reranker import build_reranking_retriever
from src.monitoring import (
    log_event, metrics, stage_timer, feedback_store, get_callbacks,
    record_retrieval_scores, extract_token_usage, record_cost,
)

load_dotenv()

# Toggle cross-encoder reranking with USE_RERANKER=false to compare against baseline.
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

app = FastAPI()

# CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load embeddings and connect to Pinecone
print("Loading embeddings...")
embedding = download_hugging_face_embeddings()
index_name = "medical-chatbot"

print("Connecting to Pinecone...")
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

TOP_N = int(os.getenv("TOP_N", "3"))          # chunks sent to the LLM (recall lever)
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))  # wide pool the reranker scores
if USE_RERANKER:
    # Retrieve a WIDE candidate set, then cross-encoder reranks to TOP_N.
    print("Building cross-encoder reranking retriever...")
    base_retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": CANDIDATE_K}
    )
    retriever = build_reranking_retriever(base_retriever, top_n=TOP_N)
else:
    # Baseline: plain similarity retrieval (no reranking).
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_N}
    )

# Request models
class ChatRequest(BaseModel):
    message: str
    provider: str = "groq"

class FeedbackRequest(BaseModel):
    rating: str          # "up" or "down"
    message: str = ""
    answer: str = ""

# LLM selector
def get_llm(provider: str):
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,  # deterministic, grounded answers -> higher faithfulness
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

# Chat endpoint (instrumented for observability)
@app.post("/chat")
async def chat(request: ChatRequest):
    provider = request.provider
    metrics.inc("rag_requests_total", provider=provider)
    callbacks = get_callbacks()
    cfg = {"callbacks": callbacks} if callbacks else None
    try:
        with stage_timer("total", provider=provider) as total_t:
            llm = get_llm(provider)
            question_answer_chain = create_stuff_documents_chain(llm, prompt)

            # --- Stage 1: RETRIEVAL latency (vector search + rerank) ---
            with stage_timer("retrieval", provider=provider) as retr_t:
                context_docs = retriever.invoke(request.message, config=cfg)

            # Retrieval relevance scores -> record so trends are visible in /metrics
            score_stats = record_retrieval_scores(context_docs, provider=provider)

            # --- Stage 2: GENERATION latency (LLM response) ---
            with stage_timer("generation", provider=provider) as gen_t:
                gen = question_answer_chain.invoke(
                    {"input": request.message, "context": context_docs}, config=cfg
                )
            answer = gen if isinstance(gen, str) else getattr(gen, "content", str(gen))

        # --- Token usage & cost per query ---
        usage = extract_token_usage(gen, request.message, context_docs, answer)
        cost = record_cost(provider, usage)

        log_event(
            "chat_completed",
            provider=provider,
            total_latency_s=round(total_t["elapsed"], 3),
            retrieval_latency_s=round(retr_t["elapsed"], 3),
            generation_latency_s=round(gen_t["elapsed"], 3),
            n_context_docs=len(context_docs),
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            cost_usd=cost,
            reranker=USE_RERANKER,
            **score_stats,
        )
        return {
            "answer": answer,
            "provider": provider,
            "total_latency_s": round(total_t["elapsed"], 3),
            "retrieval_latency_s": round(retr_t["elapsed"], 3),
            "generation_latency_s": round(gen_t["elapsed"], 3),
            "n_context_docs": len(context_docs),
            "tokens": usage,
            "cost_usd": cost,
            "reranker": USE_RERANKER,
            **score_stats,
        }
    except Exception as e:
        metrics.inc("rag_errors_total", provider=provider)
        log_event("chat_error", provider=provider, error=str(e))
        raise


# Feedback endpoint (thumbs up/down) for answer-quality tracking
@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    feedback_store.record(request.rating, request.message, request.answer)
    return {"status": "recorded", **feedback_store.summary()}


# Prometheus-style metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return PlainTextResponse(metrics.render())


# Health check
@app.get("/health")
async def health():
    return {"status": "ok", "reranker": USE_RERANKER, "feedback": feedback_store.summary()}

# Serve React frontend in production
if os.path.exists("build"):
    app.mount("/static", StaticFiles(directory="build/static"), name="static")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        return FileResponse("build/index.html")
else:
    @app.get("/")
    async def root():
        return {"message": "Medical Chatbot API is running!"}