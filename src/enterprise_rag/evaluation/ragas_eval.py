import asyncio
import logging
import math
import os
from collections.abc import Callable
from statistics import mean
from typing import Any

from enterprise_rag.api import routes
from enterprise_rag.retrieval.embedder import embed_async
from enterprise_rag.storage.postgres import authorized_chunk_ids

logger = logging.getLogger(__name__)


def _load_ragas_components() -> tuple[Any, Callable[..., Any], list[Any]]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            NoiseSensitivity,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation dependencies could not be imported. "
            "Check the installed ragas/langchain-community versions."
        ) from exc

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        NoiseSensitivity(),
    ]
    return Dataset, evaluate, metrics


async def run_pipeline(query: str, tenant_id: str, user_id: str) -> dict[str, Any]:
    """Run the retrieval and generation path used by the query API."""
    try:
        processed = await routes.preprocessor.process(query)
    except Exception:
        logger.exception("Preprocessing failed; using original query")
        processed = {
            "search_query": query,
            "use_hyde": False,
            "expanded_terms": [],
        }

    query_embedding = await embed_async(processed["search_query"])
    hits = await routes.hybrid.search(
        query=processed["search_query"],
        query_embedding=query_embedding,
        tenant_id=tenant_id,
        user_id=user_id,
        top_k=20,
    )
    if not hits:
        return {"answer": "No relevant information found.", "contexts": []}

    graph_contexts: list[dict[str, Any]] = []
    try:
        entities = await routes.graph.extract_entities(query, tenant_id)
        if entities:
            relationships = await routes.graph.search(
                entities, tenant_id, max_hops=2
            )
            graph_contexts = [
                {
                    "chunk_id": f"graph:{index}",
                    "content": (
                        f"{relationship['source']} "
                        f"-[{relationship['relationship']}]-> "
                        f"{relationship['target']}"
                    ),
                }
                for index, relationship in enumerate(relationships)
            ]
    except Exception:
        logger.exception("Graph retrieval failed during evaluation; skipping graph")

    candidates = hits + graph_contexts
    try:
        reranked = await asyncio.to_thread(
            routes._rerank_sync, processed["search_query"], candidates, 5
        )
    except Exception:
        logger.exception("Reranking failed during evaluation; using retrieval order")
        reranked = candidates[:5]

    permitted_ids = await authorized_chunk_ids(
        [str(chunk["chunk_id"]) for chunk in reranked],
        tenant_id=tenant_id,
        user_id=user_id,
    )
    contexts = [
        chunk
        for chunk in reranked
        if str(chunk["chunk_id"]).startswith("graph:")
        or str(chunk["chunk_id"]) in permitted_ids
    ]
    if not contexts:
        return {"answer": "No relevant information found.", "contexts": []}

    answer, _citations = await routes.generator.generate(query, contexts)
    return {
        "answer": answer,
        "contexts": [chunk["content"] for chunk in contexts],
    }


async def evaluate_rag(
    golden: list[dict[str, str]],
    tenant_id: str,
    user_id: str,
) -> dict[str, float | int]:
    """Evaluate the RAG pipeline against query/ground-truth examples."""
    if not golden:
        raise ValueError("golden dataset must contain at least one example")
    for index, item in enumerate(golden):
        if not isinstance(item.get("query"), str) or not item["query"].strip():
            raise ValueError(f"golden example {index} requires a non-empty query")
        if not isinstance(item.get("ground_truth"), str) or not item["ground_truth"].strip():
            raise ValueError(
                f"golden example {index} requires a non-empty ground_truth"
            )

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for the default RAGAS LLM judge."
        )

    Dataset, evaluate, metrics = _load_ragas_components()

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []
    failed = 0

    for item in golden:
        query = item["query"]
        try:
            output = await run_pipeline(query, tenant_id, user_id)
        except Exception:
            failed += 1
            logger.exception("Pipeline failed for evaluation query %r", query)
            continue

        questions.append(query)
        answers.append(output["answer"])
        contexts.append(output["contexts"])
        ground_truths.append(item["ground_truth"])

    if not questions:
        raise RuntimeError("No golden dataset examples completed successfully.")

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )
    result = await asyncio.to_thread(
        evaluate, dataset, metrics=metrics, show_progress=False
    )
    scores = _summarize_result(result)
    scores["samples_evaluated"] = len(questions)
    scores["samples_failed"] = failed
    return scores


def _summarize_result(result: Any) -> dict[str, float]:
    """Convert per-row RAGAS scores into JSON-friendly metric means."""
    if hasattr(result, "to_pandas"):
        raw_scores = result.to_pandas().to_dict(orient="list")
    elif hasattr(result, "scores"):
        raw_scores = result.scores
    else:
        raw_scores = dict(result)

    if isinstance(raw_scores, list):
        columns: dict[str, list[Any]] = {}
        for row in raw_scores:
            if isinstance(row, dict):
                for name, value in row.items():
                    columns.setdefault(name, []).append(value)
        raw_scores = columns

    summary: dict[str, float] = {}
    for name, values in raw_scores.items():
        if not isinstance(values, (list, tuple)):
            values = [values]
        numeric_values = []
        for value in values:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                numeric_values.append(numeric)
        if numeric_values:
            summary[str(name)] = mean(numeric_values)
    return summary
