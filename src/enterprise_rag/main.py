from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from enterprise_rag.api.routes import generator, graph, router
from enterprise_rag.config import settings
from enterprise_rag.storage.postgres import engine, healthcheck, init_schema


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_schema()
    try:
        yield
    finally:
        await graph.close()
        await generator.close()
        await engine.dispose()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str | bool]:
    return {"status": "ok", "db": await healthcheck()}
