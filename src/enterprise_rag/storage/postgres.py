import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from enterprise_rag.config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.postgres_url,
    echo=settings.debug,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def session() -> AsyncIterator[AsyncSession]:
    """Context manager for one session; closes it when the block exits."""
    async with SessionLocal() as s:
        yield s


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id     TEXT NOT NULL,
    title         TEXT NOT NULL,
    source        TEXT,
    metadata      JSONB DEFAULT '{}',
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id    UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    tenant_id      TEXT NOT NULL,
    content        TEXT NOT NULL,
    embedding      vector(1024),
    is_public      BOOLEAN NOT NULL DEFAULT false,
    allowed_users  TEXT[] NOT NULL DEFAULT '{}',
    chunk_index    INT NOT NULL DEFAULT 0,
    metadata       JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_chunks_tenant
    ON chunks (tenant_id);
CREATE INDEX IF NOT EXISTS idx_chunks_allowed_users
    ON chunks USING GIN (allowed_users);
CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON chunks USING GIN (to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


async def init_schema() -> None:
    """Create the extension, tables, and indexes; call once at startup."""
    async with engine.begin() as conn:
        for statement in SCHEMA_SQL.split(";"):
            if statement.strip():
                await conn.execute(text(statement))


async def healthcheck() -> bool:
    """Check whether the database is reachable."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("PostgreSQL healthcheck failed")
        return False


async def authorized_chunk_ids(
    chunk_ids: list[str], tenant_id: str, user_id: str
) -> set[str]:
    """Re-check tenant and user access for retrieved chunk IDs."""
    valid_ids: list[str] = []
    for chunk_id in chunk_ids:
        try:
            valid_ids.append(str(UUID(chunk_id)))
        except ValueError:
            continue
    if not valid_ids:
        return set()

    statement = text("""
        SELECT chunk_id
        FROM chunks
        WHERE chunk_id = ANY(CAST(:chunk_ids AS uuid[]))
          AND tenant_id = :tenant_id
          AND (is_public = true OR :user_id = ANY(allowed_users))
    """)
    async with session() as db_session:
        result = await db_session.execute(
            statement,
            {"chunk_ids": valid_ids, "tenant_id": tenant_id, "user_id": user_id},
        )
        return {str(chunk_id) for chunk_id in result.scalars()}
