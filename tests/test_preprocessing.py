import pytest

from enterprise_rag.retrieval.preprocessing import Preprocessor


@pytest.mark.asyncio
async def test_process_uses_hyde_for_short_queries_without_an_api_client() -> None:
    preprocessor = Preprocessor()
    preprocessor.client = None

    result = await preprocessor.process("refund policy")

    assert result == {
        "original_query": "refund policy",
        "search_query": "refund policy",
        "use_hyde": True,
        "expanded_terms": [],
    }


@pytest.mark.parametrize(
    "query",
    [
        "What is the exact refund policy for digital products purchased after 2024-01-01 with receipt #A1234?",
        "Tell me about RAG and ACL in the HR system",
    ],
)
def test_should_not_use_hyde_for_long_queries(query: str) -> None:
    assert Preprocessor._should_use_hyde(query) is False


def test_expand_terms_finds_acronyms_case_insensitively() -> None:
    assert Preprocessor._expand_terms("Tell me about RAG and ACL in the HR system") == [
        "retrieval augmented generation",
        "access control list",
        "human resources",
    ]
