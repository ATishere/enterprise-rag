import asyncio
from uuid import UUID, uuid5

from sqlalchemy import text

from enterprise_rag.retrieval.embedder import embed_async
from enterprise_rag.storage.postgres import engine, init_schema, session

TENANT_ID = "tenantA"
USER_ID = "userA"
DATASET_NAMESPACE = UUID("c4e4ddc4-ef55-4af3-8507-92d5d458e06e")
EVAL_DOCUMENTS = [
    (
        "refund-policy",
        "Refund policy",
        "Refunds are available within 30 days of purchase.",
    ),
    (
        "digital-products",
        "Digital product policy",
        "Digital products are non-refundable.",
    ),
    (
        "shipping-time",
        "Shipping policy",
        "Shipping takes 3-5 business days.",
    ),
]


async def seed_eval_data() -> None:
    await init_schema()
    embeddings = await asyncio.gather(
        *(embed_async(content) for _, _, content in EVAL_DOCUMENTS)
    )
    async with session() as db:
        for chunk_index, ((key, title, content), embedding) in enumerate(
            zip(EVAL_DOCUMENTS, embeddings, strict=True)
        ):
            document_id = str(uuid5(DATASET_NAMESPACE, f"document:{key}"))
            chunk_id = str(uuid5(DATASET_NAMESPACE, f"chunk:{key}"))
            await db.execute(
                text("""
                    INSERT INTO documents (document_id, tenant_id, title, source)
                    VALUES (:document_id, :tenant_id, :title, :source)
                    ON CONFLICT (document_id) DO UPDATE SET
                        tenant_id = EXCLUDED.tenant_id,
                        title = EXCLUDED.title,
                        source = EXCLUDED.source
                """),
                {
                    "document_id": document_id,
                    "tenant_id": TENANT_ID,
                    "title": title,
                    "source": f"eval:{key}",
                },
            )
            await db.execute(
                text("""
                    INSERT INTO chunks (
                        chunk_id, document_id, tenant_id, content, embedding,
                        is_public, allowed_users, chunk_index, metadata
                    )
                    VALUES (
                        :chunk_id, :document_id, :tenant_id, :content,
                        CAST(:embedding AS vector), true, ARRAY[:user_id],
                        :chunk_index, CAST(:metadata AS jsonb)
                    )
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        tenant_id = EXCLUDED.tenant_id,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        is_public = EXCLUDED.is_public,
                        allowed_users = EXCLUDED.allowed_users,
                        chunk_index = EXCLUDED.chunk_index,
                        metadata = EXCLUDED.metadata
                """),
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "tenant_id": TENANT_ID,
                    "content": content,
                    "embedding": str(embedding),
                    "user_id": USER_ID,
                    "chunk_index": chunk_index,
                    "metadata": '{"dataset": "golden-eval"}',
                },
            )
        await db.commit()


async def main() -> None:
    try:
        await seed_eval_data()
    finally:
        await engine.dispose()
    print(f"Seeded {len(EVAL_DOCUMENTS)} evaluation chunks for {TENANT_ID}.")


if __name__ == "__main__":
    asyncio.run(main())
