from types import SimpleNamespace

import pytest

from enterprise_rag import main


@pytest.mark.asyncio
async def test_health_reports_database_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_healthcheck() -> bool:
        return True

    monkeypatch.setattr(main, "healthcheck", fake_healthcheck)

    assert await main.health() == {"status": "ok", "db": True}


@pytest.mark.asyncio
async def test_lifespan_initializes_and_disposes_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_init_schema() -> None:
        calls.append("init")

    async def fake_close_graph() -> None:
        calls.append("graph")

    async def fake_close_generator() -> None:
        calls.append("generator")

    async def fake_dispose() -> None:
        calls.append("dispose")

    monkeypatch.setattr(main, "init_schema", fake_init_schema)
    monkeypatch.setattr(main.graph, "close", fake_close_graph)
    monkeypatch.setattr(main.generator, "close", fake_close_generator)
    monkeypatch.setattr(main, "engine", SimpleNamespace(dispose=fake_dispose))

    async with main.lifespan(main.app):
        assert calls == ["init"]

    assert calls == ["init", "graph", "generator", "dispose"]
