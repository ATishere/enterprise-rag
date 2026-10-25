import asyncio
import json
import logging
from collections.abc import AsyncIterator
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from enterprise_rag.api.schemas import Citation, QueryRequest, QueryResponse
from enterprise_rag.generation.llm import Generator
from enterprise_rag.retrieval.embedder import embed_async
from enterprise_rag.retrieval.graph import GraphRetriever
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.preprocessing import Preprocessor
from enterprise_rag.retrieval.reranker import Reranker
from enterprise_rag.storage.postgres import authorized_chunk_ids

logger = logging.getLogger(__name__)
router = APIRouter()

preprocessor = Preprocessor()
hybrid = HybridRetriever()
graph = GraphRetriever()
generator = Generator()
_reranker: Reranker | None = None
_reranker_lock = Lock()


def _rerank_sync(
    query: str, documents: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    global _reranker
    with _reranker_lock:
        if _reranker is None:
            _reranker = Reranker()
        return _reranker.rerank(query, documents, top_k=top_k)


async def _stream_with_error_events(
    query: str, contexts: list[dict[str, Any]]
) -> AsyncIterator[str]:
    try:
        async for event in generator.stream_generate(query, contexts):
            yield event
    except Exception:
        logger.exception("Streaming generation failed")
        payload = json.dumps({"error": "Generation unavailable"})
        yield f"event: error\ndata: {payload}\n\n"


@router.post("/api/v1/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse | StreamingResponse:
    trace_id = str(uuid4())
    try:
        processed = await preprocessor.process(req.query)
    except Exception:
        logger.exception("Preprocessing failed; using original query")
        processed = {
            "original_query": req.query,
            "search_query": req.query,
            "use_hyde": False,
            "expanded_terms": [],
        }

    try:
        query_embedding = await embed_async(processed["search_query"])
        hybrid_results = await hybrid.search(
            query=processed["search_query"],
            query_embedding=query_embedding,
            tenant_id=req.tenant_id,
            user_id=req.user_id,
            top_k=req.top_k * 4,
        )
    except Exception as exc:
        logger.exception("Hybrid retrieval failed (trace_id=%s)", trace_id)
        raise HTTPException(
            status_code=503, detail="Retrieval unavailable"
        ) from exc

    if not hybrid_results:
        return QueryResponse(
            answer="No relevant information found.",
            citations=[],
            context_used=0,
            hyde_used=processed["use_hyde"],
            graph_used=False,
            trace_id=trace_id,
        )

    graph_used = False
    graph_context: list[dict[str, Any]] = []
    try:
        entities = await graph.extract_entities(req.query, req.tenant_id)
        if entities:
            relationships = await graph.search(
                entities, req.tenant_id, max_hops=2
            )
            graph_context = [
                {
                    "chunk_id": f"graph:{index}",
                    "content": (
                        f"{relationship['source']} "
                        f"-[{relationship['relationship']}]-> "
                        f"{relationship['target']}"
                    ),
                }
                for index, relationship in enumerate(relationships)
            ]
            graph_used = bool(graph_context)
    except Exception:
        logger.exception("Graph retrieval failed; continuing without graph context")

    combined = hybrid_results + graph_context
    try:
        final_context = await asyncio.to_thread(
            _rerank_sync, processed["search_query"], combined, req.top_k
        )
    except Exception:
        logger.exception("Reranking failed; using retrieval order")
        final_context = combined[: req.top_k]

    try:
        permitted_ids = await authorized_chunk_ids(
            [str(item["chunk_id"]) for item in final_context],
            tenant_id=req.tenant_id,
            user_id=req.user_id,
        )
    except Exception as exc:
        logger.exception("Final ACL verification failed (trace_id=%s)", trace_id)
        raise HTTPException(
            status_code=503, detail="Authorization verification unavailable"
        ) from exc

    final_context = [
        item
        for item in final_context
        if str(item["chunk_id"]).startswith("graph:")
        or str(item["chunk_id"]) in permitted_ids
    ]
    if not final_context:
        return QueryResponse(
            answer="No relevant information found.",
            citations=[],
            context_used=0,
            hyde_used=processed["use_hyde"],
            graph_used=False,
            trace_id=trace_id,
        )

    if req.stream:
        return StreamingResponse(
            _stream_with_error_events(req.query, final_context),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        answer, citations = await generator.generate(req.query, final_context)
    except Exception as exc:
        logger.exception("Generation failed (trace_id=%s)", trace_id)
        raise HTTPException(
            status_code=503, detail="Generation unavailable"
        ) from exc

    return QueryResponse(
        answer=answer,
        citations=[
            Citation(
                text=str(citation["text"]),
                chunk_id=str(citation["chunk_id"]),
                confidence=float(citation["confidence"]),
            )
            for citation in citations
        ],
        context_used=len(final_context),
        hyde_used=processed["use_hyde"],
        graph_used=graph_used,
        trace_id=trace_id,
    )
