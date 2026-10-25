import json
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def parse_and_validate_citations(
    raw_json: str,
    answer: str,
    valid_chunk_ids: set[str],
) -> tuple[str, list[dict[str, str | float]]]:
    """Parse LLM JSON and retain only citations grounded in its answer/context."""
    try:
        data: Any = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        logger.warning("LLM returned invalid JSON for citations")
        return answer, []

    if not isinstance(data, dict):
        logger.warning("LLM returned a non-object JSON value for citations")
        return answer, []

    answer_text = data.get("answer")
    if not isinstance(answer_text, str) or not answer_text:
        answer_text = answer
    raw_citations = data.get("citations")
    if not isinstance(raw_citations, list):
        raw_citations = []

    validated: list[dict[str, str | float]] = []
    for citation in raw_citations:
        if not isinstance(citation, dict):
            continue
        span = citation.get("text")
        chunk_id = citation.get("chunk_id")
        if not isinstance(span, str) or not isinstance(chunk_id, str):
            continue

        span = span.strip()
        if not span or span not in answer_text or chunk_id not in valid_chunk_ids:
            continue

        try:
            confidence = float(citation.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not math.isfinite(confidence):
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))

        validated.append(
            {"text": span, "chunk_id": chunk_id, "confidence": confidence}
        )

    return answer_text, validated
