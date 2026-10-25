import pytest

from enterprise_rag.generation.llm import Generator


@pytest.mark.asyncio
async def test_generate_without_context_does_not_require_api_key() -> None:
    generator = Generator()
    generator.client = None

    answer, citations = await generator.generate("Question?", [])

    assert answer == "I don't have enough information to answer."
    assert citations == []


@pytest.mark.asyncio
async def test_generate_with_context_requires_api_key() -> None:
    generator = Generator()
    generator.client = None

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await generator.generate(
            "Question?", [{"chunk_id": "c1", "content": "Context"}]
        )
