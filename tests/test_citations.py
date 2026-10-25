from enterprise_rag.generation.citations import parse_and_validate_citations


def test_valid_citation_passes() -> None:
    raw = (
        '{"answer":"Refunds within 30 days.","citations":'
        '[{"text":"Refunds within 30 days","chunk_id":"c1","confidence":0.9}]}'
    )

    answer, citations = parse_and_validate_citations(raw, "fallback", {"c1"})

    assert answer == "Refunds within 30 days."
    assert len(citations) == 1
    assert citations[0]["chunk_id"] == "c1"


def test_fake_span_is_rejected() -> None:
    raw = (
        '{"answer":"Refunds within 30 days.","citations":'
        '[{"text":"Fake text","chunk_id":"c1","confidence":0.9}]}'
    )

    _, citations = parse_and_validate_citations(raw, "x", {"c1"})

    assert citations == []


def test_fake_chunk_id_is_rejected() -> None:
    raw = (
        '{"answer":"Refunds within 30 days.","citations":'
        '[{"text":"Refunds within 30 days","chunk_id":"c999","confidence":0.9}]}'
    )

    _, citations = parse_and_validate_citations(raw, "x", {"c1"})

    assert citations == []


def test_invalid_json_uses_fallback_answer() -> None:
    answer, citations = parse_and_validate_citations(
        "not json", "fallback answer", {"c1"}
    )

    assert answer == "fallback answer"
    assert citations == []
