import logging
from importlib import import_module
from typing import Any

from neo4j import AsyncGraphDatabase

from enterprise_rag.config import settings

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Neo4j Graph RAG retriever for multi-hop questions."""

    def __init__(self) -> None:
        self.driver = AsyncGraphDatabase.driver(
            settings.neo4j_url,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    async def close(self) -> None:
        await self.driver.close()

    async def extract_entities(self, query: str, tenant_id: str) -> list[str]:
        """Match known tenant entities, then fall back to spaCy NER."""
        try:
            known = await self._known_entities(tenant_id)
        except Exception:
            logger.exception("Neo4j entity lookup failed for tenant %s", tenant_id)
            return self._spacy_extract(query)

        query_lower = query.lower()
        matched = [name for name in known if name.lower() in query_lower]
        if matched:
            return matched
        return self._spacy_extract(query)

    async def _known_entities(self, tenant_id: str) -> list[str]:
        """Fetch at most 5000 entity names for the tenant."""
        cypher = """
        MATCH (e:Entity {tenant_id: $tenant_id})
        RETURN e.name AS name
        LIMIT 5000
        """
        async with self.driver.session() as neo_session:
            result = await neo_session.run(cypher, tenant_id=tenant_id)
            return [
                record["name"]
                async for record in result
                if isinstance(record["name"], str)
            ]

    @staticmethod
    def _spacy_extract(query: str) -> list[str]:
        try:
            spacy: Any = import_module("spacy")
        except ImportError:
            logger.info("spaCy is not installed; entity NER fallback is unavailable")
            return []

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model en_core_web_sm is unavailable; entity NER fallback skipped"
            )
            return []

        return [entity.text for entity in nlp(query).ents]

    async def search(
        self,
        entities: list[str],
        tenant_id: str,
        max_hops: int = 2,
        limit: int = 30,
    ) -> list[dict]:
        """Traverse tenant-isolated graph paths and return their relationships."""
        if not entities:
            return []
        if limit < 1:
            raise ValueError("limit must be greater than zero")

        try:
            hops = max(1, min(int(max_hops), 4))
        except (TypeError, ValueError) as exc:
            raise ValueError("max_hops must be an integer") from exc

        cypher = f"""
        MATCH path = (e:Entity)-[*1..{hops}]-(related:Entity)
        WHERE e.name IN $entities
          AND e.tenant_id = $tenant_id
          AND all(node IN nodes(path) WHERE node.tenant_id = $tenant_id)
        UNWIND relationships(path) AS rel
        RETURN DISTINCT
            startNode(rel).name AS source,
            type(rel) AS relationship,
            endNode(rel).name AS target
        LIMIT $limit
        """
        try:
            async with self.driver.session() as neo_session:
                result = await neo_session.run(
                    cypher,
                    entities=entities,
                    tenant_id=tenant_id,
                    limit=limit,
                )
                return [dict(record) async for record in result]
        except Exception:
            logger.exception("Graph search failed for tenant %s", tenant_id)
            return []
