import logging
import re
from typing import Any

from anthropic import APIError, AsyncAnthropic

from enterprise_rag.config import settings

logger = logging.getLogger(__name__)


class Preprocessor:
    """HyDE + query expansion — prepares queries before retrieval."""

    def __init__(self) -> None:
        self.client = (
            AsyncAnthropic(api_key=settings.anthropic_api_key)
            if settings.anthropic_api_key
            else None
        )

    async def process(self, query: str) -> dict[str, Any]:
        """Return original/search queries, HyDE status, and expanded terms."""
        use_hyde = self._should_use_hyde(query)

        search_query = query
        if use_hyde and self.client is not None:
            hypothetical = await self._generate_hypothetical(query)
            if hypothetical:
                search_query = hypothetical

        return {
            "original_query": query,
            "search_query": search_query,
            "use_hyde": use_hyde,
            "expanded_terms": self._expand_terms(query),
        }

    @staticmethod
    def _should_use_hyde(query: str) -> bool:
        """Use HyDE for short queries without specific numeric identifiers."""
        if len(query.split()) < 5:
            return True
        if re.search(r"\b\d+\b|[A-Z]{2,}-\d+", query):
            return False
        return False

    async def _generate_hypothetical(self, query: str) -> str | None:
        """Ask the LLM for a short hypothetical passage to use in retrieval."""
        prompt = (
            "Write a short, factual paragraph (3-5 sentences) that would "
            "directly answer the following question. Write as if you are a "
            "knowledge base document. Do not say 'I think' or 'maybe'. "
            f"Question: {query}"
        )
        client = self.client
        if client is None:
            return None
        try:
            response = await client.messages.create(
                model=settings.llm_model,
                max_tokens=256,
                extra_body={"temperature": 0.0},
                messages=[{"role": "user", "content": prompt}],
            )
        except APIError:
            logger.exception("Anthropic HyDE request failed; using original query")
            return None

        if not response.content:
            logger.warning("Anthropic HyDE response was empty; using original query")
            return None

        hypothetical = next(
            (
                block.text.strip()
                for block in response.content
                if hasattr(block, "text")
            ),
            "",
        )
        return hypothetical or None

    @staticmethod
    def _expand_terms(query: str) -> list[str]:
        """Expand common acronyms without an LLM request."""
        acronyms = {
            "llm": ["large language model"],
            "rag": ["retrieval augmented generation"],
            "acl": ["access control list"],
            "hr": ["human resources"],
            "api": ["application programming interface"],
        }
        found: list[str] = []
        lower = query.lower()
        for acronym, expansions in acronyms.items():
            if re.search(rf"\b{acronym}\b", lower):
                found.extend(expansions)
        return found
