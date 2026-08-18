import asyncio
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

from .config import Settings
from .observability import timed


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    model: str
    latency_ms: float
    estimated_cost_usd: float


class GeminiEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        from google import genai

        self.settings = settings
        if not settings.google_cloud_project:
            raise ValueError("GOOGLE_CLOUD_PROJECT is required for Vertex AI embeddings")
        self.client = genai.Client(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )

    async def embed(self, texts: Sequence[str], *, task_type: str) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult([], self.settings.embedding_model, 0.0, 0.0)
        finish = timed("embedding")

        def call() -> object:
            from google.genai.types import EmbedContentConfig

            return self.client.models.embed_content(
                model=self.settings.embedding_model,
                contents=list(texts),
                config=EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.settings.embedding_dimension,
                ),
            )

        response = await asyncio.to_thread(call)
        embeddings = getattr(response, "embeddings", None) or []
        vectors = [list(getattr(item, "values", []) or []) for item in embeddings]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding response count {len(vectors)} != input count {len(texts)}"
            )
        usage = finish(model=self.settings.embedding_model, cost=0.0)
        return EmbeddingResult(
            vectors, self.settings.embedding_model, usage.latency_ms, usage.estimated_cost_usd
        )


class LocalHashEmbedder:
    """Deterministic offline embedding for the local demo and unit tests only."""

    def __init__(self, dimension: int = 32) -> None:
        self.dimension = dimension

    async def embed(self, texts: Sequence[str], *, task_type: str) -> EmbeddingResult:
        del task_type
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                vector[index] += 1.0 if digest[4] % 2 else -1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return EmbeddingResult(vectors, "local-hash-demo", 0.0, 0.0)
