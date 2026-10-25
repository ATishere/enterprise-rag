import json
import logging
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from enterprise_rag.config import settings
from enterprise_rag.generation.citations import parse_and_validate_citations

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise knowledge-base assistant.
Answer ONLY using the provided context. If the context does not contain
the answer, say so explicitly.

Return STRICT JSON with this schema:
{
  "answer": "<your answer as plain text>",
  "citations": [
    {"text": "<exact substring copied from your answer>",
     "chunk_id": "<id of the source chunk>",
     "confidence": <float 0..1>}
  ]
}
Rules:
- Each "text" MUST be an exact substring of "answer".
- Each "chunk_id" MUST be one of the provided context chunk ids.
- Do not invent citations.
"""


def _format_context(contexts: list[dict]) -> str:
    """Format contexts with chunk identifiers for citation generation."""
    return "\n\n".join(
        f"[chunk_id={context['chunk_id']}]\n{context['content']}"
        for context in contexts
    )


class Generator:
    """Generate answers and validated citations from retrieved context."""

    def __init__(self) -> None:
        self.client = (
            AsyncAnthropic(api_key=settings.anthropic_api_key)
            if settings.anthropic_api_key
            else None
        )

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()

    def _require_client(self) -> AsyncAnthropic:
        if self.client is None:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required to generate answers."
            )
        return self.client

    async def generate(
        self, query: str, contexts: list[dict]
    ) -> tuple[str, list[dict[str, str | float]]]:
        """Return an answer and citations validated against the supplied context."""
        if not contexts:
            return "I don't have enough information to answer.", []

        client = self._require_client()
        valid_ids = {str(context["chunk_id"]) for context in contexts}
        user_message = (
            f"Context:\n{_format_context(contexts)}\n\n"
            f"Question: {query}\n\n"
            "Answer in the JSON schema from the system prompt."
        )

        response = await client.messages.create(
            model=settings.llm_model,
            max_tokens=1024,
            extra_body={"temperature": 0.0},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = next(
            (block.text.strip() for block in response.content if hasattr(block, "text")),
            "",
        )
        if not raw:
            logger.warning("Anthropic returned an empty answer")
            return "I couldn't generate a valid response.", []

        return parse_and_validate_citations(
            raw,
            answer="I couldn't generate a valid response.",
            valid_chunk_ids=valid_ids,
        )

    async def stream_generate(
        self, query: str, contexts: list[dict]
    ) -> AsyncIterator[str]:
        """Stream answer deltas as server-sent events; citations are not included."""
        if not contexts:
            message = "I don't have enough information to answer."
            yield f"data: {json.dumps({'delta': message})}\n\n"
            yield "data: [DONE]\n\n"
            return

        client = self._require_client()
        user_message = f"Context:\n{_format_context(contexts)}\n\nQuestion: {query}"

        async with client.messages.stream(
            model=settings.llm_model,
            max_tokens=1024,
            extra_body={"temperature": 0.0},
            system="Answer concisely using only the context.",
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for delta in stream.text_stream:
                yield f"data: {json.dumps({'delta': delta})}\n\n"
        yield "data: [DONE]\n\n"
