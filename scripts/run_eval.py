import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from enterprise_rag.api.routes import generator, graph
from enterprise_rag.evaluation.ragas_eval import evaluate_rag
from enterprise_rag.storage.postgres import engine

TENANT = "tenantA"
USER = "userA"
HARD_GATES = {"faithfulness": 0.90}
SOFT_GATES = {
    "answer_relevancy": 0.85,
    "context_precision": 0.80,
    "context_recall": 0.75,
}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-gate", choices=HARD_GATES, default="faithfulness")
    args = parser.parse_args()

    dataset_path = (
        Path(__file__).resolve().parents[1] / "tests" / "golden_dataset.json"
    )
    golden: Any = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(golden, list):
        raise TypeError(f"{dataset_path} must contain a JSON array")

    try:
        result = await evaluate_rag(golden, TENANT, USER)
    finally:
        await graph.close()
        await generator.close()
        await engine.dispose()
    print(json.dumps(result, indent=2, default=float))

    if result.get("samples_failed", 0):
        print(f"::error:: Pipeline failed for {result['samples_failed']} eval samples")
        raise SystemExit(1)

    hard_failed = [
        f"{metric}={result.get(metric, 0.0):.3f} < {threshold}"
        for metric, threshold in HARD_GATES.items()
        if metric == args.hard_gate
        and result.get(metric, 0.0) < threshold
    ]
    soft_failed = [
        f"{metric}={result.get(metric, 0.0):.3f} < {threshold}"
        for metric, threshold in SOFT_GATES.items()
        if result.get(metric, 0.0) < threshold
    ]

    if soft_failed:
        print(f"::warning:: Soft gate below threshold: {', '.join(soft_failed)}")
    if hard_failed:
        print(f"::error:: Hard gate FAILED: {', '.join(hard_failed)}")
        raise SystemExit(1)
    print(f"Hard gate PASSED: {args.hard_gate}")


if __name__ == "__main__":
    asyncio.run(main())
