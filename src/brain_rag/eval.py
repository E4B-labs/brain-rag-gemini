from collections.abc import Sequence
from dataclasses import dataclass

from .models import FactChunk


@dataclass(frozen=True)
class GoldCase:
    question: str
    relevant_fact_ids: frozenset[str]
    expected_answer: str


def recall_at_k(retrieved: Sequence[FactChunk], relevant_fact_ids: frozenset[str], k: int) -> float:
    if not relevant_fact_ids:
        return 1.0
    found = {chunk.fact_id for chunk in retrieved[:k]}
    return len(found & relevant_fact_ids) / len(relevant_fact_ids)


def faithfulness(
    answer: str, citations: Sequence[str], available_fact_ids: frozenset[str]
) -> float:
    del answer
    return 1.0 if citations and set(citations) <= available_fact_ids else 0.0
