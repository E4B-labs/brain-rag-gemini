from collections.abc import Sequence
from typing import Protocol

from .embeddings import EmbeddingResult
from .generation import GenerationResult
from .models import BrainFact, FactChunk, ScoredChunk


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str], *, task_type: str) -> EmbeddingResult: ...


class Generator(Protocol):
    async def generate(
        self,
        question: str,
        contexts: Sequence[FactChunk],
        *,
        max_output_tokens: int,
    ) -> GenerationResult: ...


class BrainSource(Protocol):
    async def fetch_facts(
        self, workspace_id: str, *, entity: str | None = None, section: str | None = None
    ) -> list[BrainFact]: ...


class VectorStore(Protocol):
    async def upsert(
        self, chunks: Sequence[FactChunk], vectors: Sequence[Sequence[float]]
    ) -> int: ...

    async def search(
        self,
        query_vector: Sequence[float],
        query_text: str,
        *,
        top_k: int,
        entity: str | None = None,
        section: str | None = None,
    ) -> list[ScoredChunk]: ...
