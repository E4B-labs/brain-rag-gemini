from collections.abc import Sequence

from .models import ScoredChunk


def rerank(
    results: Sequence[ScoredChunk],
    *,
    entity: str | None = None,
    section: str | None = None,
    top_k: int,
) -> list[ScoredChunk]:
    reranked: list[ScoredChunk] = []
    for item in results:
        boost = 0.0
        if entity and entity.lower() in item.chunk.entity_name.lower():
            boost += 0.12
        if section and section == item.chunk.section:
            boost += 0.08
        score = item.score + boost
        reranked.append(item.model_copy(update={"score": score}))
    reranked.sort(
        key=lambda item: (item.score, item.lexical_score, item.chunk.fact_id), reverse=True
    )
    return reranked[:top_k]
