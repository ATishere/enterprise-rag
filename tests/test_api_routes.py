from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from enterprise_rag.api import routes
from enterprise_rag.api.schemas import QueryRequest, QueryResponse


class StubPreprocessor:
    async def process(self, query: str) -> dict[str, Any]:
        return {
            "original_query": query,
            "search_query": query,
            "use_hyde": False,
            "expanded_terms": [],
        }


class StubHybrid:
    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self.results = results or []

    async def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.results


class StubGraph:
    async def extract_entities(self, query: str, tenant_id: str) -> list[str]:
        return []

    async def search(self, *args: Any, **kwargs: Any) -> list[dict[str, str]]:
        return []


class StubGenerator:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self, query: str, contexts: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        self.calls += 1
        return "Grounded answer.", []

    async def stream_generate(self, query: str, contexts: list[dict[str, Any]]):
        raise RuntimeError("stream request failed")
        yield ""


@pytest.fixture
def query_request() -> QueryRequest:
    return QueryRequest(query="refund policy", tenant_id="tenantA", user_id="userA")


@pytest.mark.asyncio
async def test_empty_retrieval_does_not_call_generator(
    monkeypatch: pytest.MonkeyPatch, query_request: QueryRequest
) -> None:
    generator = StubGenerator()
    monkeypatch.setattr(routes, "preprocessor", StubPreprocessor())
    monkeypatch.setattr(routes, "embed_async", _fake_embed)
    monkeypatch.setattr(routes, "hybrid", StubHybrid())
    monkeypatch.setattr(routes, "generator", generator)

    response = await routes.query(query_request)

    assert isinstance(response, QueryResponse)
    assert response.answer == "No relevant information found."
    assert generator.calls == 0


@pytest.mark.asyncio
async def test_final_acl_check_removes_revoked_chunk(
    monkeypatch: pytest.MonkeyPatch, query_request: QueryRequest
) -> None:
    chunks = [
        {"chunk_id": "allowed-id", "content": "public context"},
        {"chunk_id": "revoked-id", "content": "private context"},
    ]
    generator = StubGenerator()
    monkeypatch.setattr(routes, "preprocessor", StubPreprocessor())
    monkeypatch.setattr(routes, "embed_async", _fake_embed)
    monkeypatch.setattr(routes, "hybrid", StubHybrid(chunks))
    monkeypatch.setattr(routes, "graph", StubGraph())
    monkeypatch.setattr(routes, "_rerank_sync", lambda query, docs, top_k: docs)
    monkeypatch.setattr(
        routes, "authorized_chunk_ids", _authorized_only_allowed
    )
    monkeypatch.setattr(routes, "generator", generator)

    response = await routes.query(query_request)

    assert isinstance(response, QueryResponse)
    assert response.context_used == 1
    assert generator.calls == 1


@pytest.mark.asyncio
async def test_retrieval_failure_returns_503(
    monkeypatch: pytest.MonkeyPatch, query_request: QueryRequest
) -> None:
    class FailedHybrid:
        async def search(self, **kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(routes, "preprocessor", StubPreprocessor())
    monkeypatch.setattr(routes, "embed_async", _fake_embed)
    monkeypatch.setattr(routes, "hybrid", FailedHybrid())

    with pytest.raises(HTTPException) as exc_info:
        await routes.query(query_request)

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_graph_failure_degrades_to_hybrid_results(
    monkeypatch: pytest.MonkeyPatch, query_request: QueryRequest
) -> None:
    class FailedGraph:
        async def extract_entities(self, query: str, tenant_id: str) -> list[str]:
            raise RuntimeError("Neo4j unavailable")

    monkeypatch.setattr(routes, "preprocessor", StubPreprocessor())
    monkeypatch.setattr(routes, "embed_async", _fake_embed)
    monkeypatch.setattr(
        routes, "hybrid", StubHybrid([{"chunk_id": "chunk-id", "content": "ctx"}])
    )
    monkeypatch.setattr(routes, "graph", FailedGraph())
    monkeypatch.setattr(routes, "_rerank_sync", lambda query, docs, top_k: docs)
    monkeypatch.setattr(routes, "authorized_chunk_ids", _authorized_all)
    monkeypatch.setattr(routes, "generator", StubGenerator())

    response = await routes.query(query_request)

    assert isinstance(response, QueryResponse)
    assert response.answer == "Grounded answer."
    assert response.graph_used is False


@pytest.mark.asyncio
async def test_generation_failure_returns_503(
    monkeypatch: pytest.MonkeyPatch, query_request: QueryRequest
) -> None:
    class FailedGenerator:
        async def generate(
            self, query: str, contexts: list[dict[str, Any]]
        ) -> tuple[str, list[dict[str, Any]]]:
            raise RuntimeError("API key missing")

    monkeypatch.setattr(routes, "preprocessor", StubPreprocessor())
    monkeypatch.setattr(routes, "embed_async", _fake_embed)
    monkeypatch.setattr(
        routes, "hybrid", StubHybrid([{"chunk_id": "chunk-id", "content": "ctx"}])
    )
    monkeypatch.setattr(routes, "graph", StubGraph())
    monkeypatch.setattr(routes, "_rerank_sync", lambda query, docs, top_k: docs)
    monkeypatch.setattr(routes, "authorized_chunk_ids", _authorized_all)
    monkeypatch.setattr(routes, "generator", FailedGenerator())

    with pytest.raises(HTTPException) as exc_info:
        await routes.query(query_request)

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_stream_error_is_reported_as_sse_event(
    monkeypatch: pytest.MonkeyPatch, query_request: QueryRequest
) -> None:
    monkeypatch.setattr(routes, "preprocessor", StubPreprocessor())
    monkeypatch.setattr(routes, "embed_async", _fake_embed)
    monkeypatch.setattr(
        routes, "hybrid", StubHybrid([{"chunk_id": "chunk-id", "content": "ctx"}])
    )
    monkeypatch.setattr(routes, "graph", StubGraph())
    monkeypatch.setattr(routes, "_rerank_sync", lambda query, docs, top_k: docs)
    monkeypatch.setattr(routes, "authorized_chunk_ids", _authorized_all)
    monkeypatch.setattr(routes, "generator", StubGenerator())
    query_request.stream = True

    response = await routes.query(query_request)

    assert isinstance(response, StreamingResponse)
    events = [event async for event in response.body_iterator]
    assert any("event: error" in event for event in events)
    assert any("Generation unavailable" in event for event in events)


async def _fake_embed(text: str) -> list[float]:
    return [0.0] * 1024


async def _authorized_only_allowed(
    chunk_ids: list[str], tenant_id: str, user_id: str
) -> set[str]:
    return {"allowed-id"}


async def _authorized_all(
    chunk_ids: list[str], tenant_id: str, user_id: str
) -> set[str]:
    return set(chunk_ids)
