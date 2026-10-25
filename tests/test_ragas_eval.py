import pytest

from enterprise_rag.evaluation import ragas_eval


class FakeDataset:
    @staticmethod
    def from_dict(data: dict[str, list]) -> dict[str, list]:
        return data


@pytest.mark.asyncio
async def test_evaluate_rag_averages_metric_scores_and_counts_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_pipeline(query: str, tenant_id: str, user_id: str) -> dict:
        return {"answer": f"answer for {query}", "contexts": ["retrieved context"]}

    def fake_evaluate(dataset: dict, metrics: list, show_progress: bool) -> dict:
        assert dataset["question"] == ["Q1", "Q2"]
        assert dataset["ground_truth"] == ["GT1", "GT2"]
        assert len(metrics) == 5
        assert show_progress is False
        return {
            "faithfulness": [1.0, 0.8],
            "answer_relevancy": [0.9, 0.7],
        }

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(ragas_eval, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(
        ragas_eval,
        "_load_ragas_components",
        lambda: (FakeDataset, fake_evaluate, [object()] * 5),
    )

    result = await ragas_eval.evaluate_rag(
        [
            {"query": "Q1", "ground_truth": "GT1"},
            {"query": "Q2", "ground_truth": "GT2"},
        ],
        tenant_id="tenantA",
        user_id="userA",
    )

    assert result == {
        "faithfulness": pytest.approx(0.9),
        "answer_relevancy": pytest.approx(0.8),
        "samples_evaluated": 2,
        "samples_failed": 0,
    }


@pytest.mark.asyncio
async def test_evaluate_rag_rejects_missing_openai_judge_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await ragas_eval.evaluate_rag(
            [{"query": "Q", "ground_truth": "GT"}], "tenantA", "userA"
        )


@pytest.mark.asyncio
async def test_evaluate_rag_rejects_empty_golden_data() -> None:
    with pytest.raises(ValueError, match="at least one"):
        await ragas_eval.evaluate_rag([], "tenantA", "userA")


def test_summarize_result_ignores_non_finite_scores() -> None:
    result = ragas_eval._summarize_result(
        {"faithfulness": [0.5, float("nan"), 1.0], "empty": [float("inf")]}
    )

    assert result == {"faithfulness": pytest.approx(0.75)}
