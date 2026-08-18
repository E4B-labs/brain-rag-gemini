from collections.abc import Sequence

import pytest

from brain_rag.embeddings import EmbeddingResult
from brain_rag.generation import GenerationResult
from brain_rag.models import AnswerPayload, BrainFact


class FakeEmbedder:
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.calls: list[tuple[list[str], str]] = []

    async def embed(self, texts: Sequence[str], *, task_type: str) -> EmbeddingResult:
        self.calls.append((list(texts), task_type))
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("tasktree" in lowered or "firestore" in lowered),
                    float("brain" in lowered or "supabase" in lowered),
                    float("deployment" in lowered or "wdrozenie" in lowered),
                    float(len(lowered.split()) % 5),
                ][: self.dimension]
            )
        return EmbeddingResult(vectors, "fake-embedding", 0.01, 0.0)


class FakeGenerator:
    def __init__(self, citation_id: str = "f-1", cost: float = 0.0) -> None:
        self.citation_id = citation_id
        self.cost = cost
        self.calls: list[tuple[str, list[str], int]] = []

    async def generate(self, question, contexts, *, max_output_tokens: int) -> GenerationResult:
        self.calls.append((question, [context.fact_id for context in contexts], max_output_tokens))
        return GenerationResult(
            AnswerPayload(answer="Grounded fake answer", citations=[self.citation_id]),
            "fake-gemini",
            0.02,
            self.cost,
        )


@pytest.fixture
def facts() -> list[BrainFact]:
    return [
        BrainFact(
            fact_id="f-1",
            entity_id="e-tasktree",
            entity_kind="project",
            entity_name="TaskTree",
            body="TaskTree stores operational data in Firestore.",
            section="stack",
        ),
        BrainFact(
            fact_id="f-2",
            entity_id="e-brain",
            entity_kind="concept",
            entity_name="Brain",
            body="Brain stores entities and observations in Supabase Postgres.",
            section="wdrozenie",
        ),
        BrainFact(
            fact_id="f-3",
            entity_id="e-brain",
            entity_kind="concept",
            entity_name="Brain",
            body="The production vector index is maintained separately.",
            section="stan",
        ),
    ]

