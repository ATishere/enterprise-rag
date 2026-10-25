from sentence_transformers import CrossEncoder

from enterprise_rag.config import settings


class Reranker:
    """Cross-encoder reranker that scores each query-document pair jointly."""

    def __init__(self) -> None:
        self.model = CrossEncoder(settings.reranker_model, max_length=512)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Score documents against the query and return the top_k results."""
        if not documents:
            return []
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        if top_k == 0:
            return []

        pairs = [[query, doc["content"]] for doc in documents]
        scores = self.model.predict(
            pairs,
            batch_size=16,
            truncation=True,
            show_progress_bar=False,
        )

        reranked = []
        for doc, score in zip(documents, scores, strict=True):
            reranked.append({**doc, "rerank_score": float(score)})

        reranked.sort(key=lambda doc: doc["rerank_score"], reverse=True)
        return reranked[:top_k]
