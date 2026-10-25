from sqlalchemy import text

from enterprise_rag.storage.postgres import session


class HybridRetriever:
    """BM25 + vector search with RRF fusion and SQL-enforced ACL filtering."""

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        tenant_id: str,
        user_id: str,
        top_k: int = 20,
    ) -> list[dict]:
        """Return top_k chunks fused from vector and full-text search."""
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        if len(query_embedding) != 1024:
            raise ValueError("query_embedding must contain exactly 1024 values")

        sql = text("""
            WITH vector_search AS (
                SELECT
                    chunk_id, document_id, content, metadata,
                    1 - (embedding <=> CAST(:embedding AS vector)) AS vector_score,
                    ROW_NUMBER() OVER (
                        ORDER BY embedding <=> CAST(:embedding AS vector), chunk_id
                    ) AS vector_rank
                FROM chunks
                WHERE tenant_id = :tenant_id
                  AND (is_public = true OR :user_id = ANY(allowed_users))
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:embedding AS vector), chunk_id
                LIMIT :candidate_limit
            ),
            bm25_search AS (
                SELECT
                    chunk_id, document_id, content, metadata,
                    ts_rank(
                        to_tsvector('english', content),
                        plainto_tsquery('english', :query)
                    ) AS bm25_score,
                    ROW_NUMBER() OVER (
                        ORDER BY
                            ts_rank(
                                to_tsvector('english', content),
                                plainto_tsquery('english', :query)
                            ) DESC,
                            chunk_id
                    ) AS bm25_rank
                FROM chunks
                WHERE tenant_id = :tenant_id
                  AND (is_public = true OR :user_id = ANY(allowed_users))
                  AND to_tsvector('english', content)
                      @@ plainto_tsquery('english', :query)
                ORDER BY
                    ts_rank(
                        to_tsvector('english', content),
                        plainto_tsquery('english', :query)
                    ) DESC,
                    chunk_id
                LIMIT :candidate_limit
            )
            SELECT
                COALESCE(v.chunk_id, b.chunk_id) AS chunk_id,
                COALESCE(v.document_id, b.document_id) AS document_id,
                COALESCE(v.content, b.content) AS content,
                COALESCE(v.metadata, b.metadata) AS metadata,
                COALESCE(v.vector_score, 0) AS vector_score,
                COALESCE(b.bm25_score, 0) AS bm25_score,
                COALESCE(1.0 / (60 + v.vector_rank), 0)
                  + COALESCE(1.0 / (60 + b.bm25_rank), 0) AS rrf_score
            FROM vector_search v
            FULL OUTER JOIN bm25_search b ON v.chunk_id = b.chunk_id
            ORDER BY rrf_score DESC
            LIMIT :top_k
        """)

        params = {
            "embedding": str(query_embedding),
            "query": query,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "candidate_limit": top_k * 3,
            "top_k": top_k,
        }

        async with session() as s:
            result = await s.execute(sql, params)
            return [dict(row._mapping) for row in result]
