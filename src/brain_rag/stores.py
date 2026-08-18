import asyncio
import json
import math
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from .models import FactChunk, ScoredChunk

TOKEN_RE = re.compile(r"[\w-]{2,}", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _lexical(query: str, text: str) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    return len(q & _tokens(text)) / len(q)


def _score(chunk: FactChunk, vector: Sequence[float], query: str) -> ScoredChunk:
    vector_score = _cosine(vector, getattr(chunk, "_vector", []))
    lexical_score = _lexical(query, f"{chunk.entity_name} {chunk.section} {chunk.text}")
    combined = max(0.0, 0.75 * ((vector_score + 1) / 2) + 0.25 * lexical_score)
    return ScoredChunk(
        chunk=chunk,
        vector_score=vector_score,
        lexical_score=lexical_score,
        score=combined,
    )


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: dict[str, tuple[FactChunk, list[float]]] = {}

    async def upsert(self, chunks: Sequence[FactChunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._items[chunk.chunk_id] = (chunk, list(vector))
        return len(chunks)

    async def search(
        self,
        query_vector: Sequence[float],
        query_text: str,
        *,
        top_k: int,
        entity: str | None = None,
        section: str | None = None,
    ) -> list[ScoredChunk]:
        results: list[ScoredChunk] = []
        for chunk, vector in self._items.values():
            if entity and entity.lower() not in chunk.entity_name.lower():
                continue
            if section and section != chunk.section:
                continue
            vector_score = _cosine(query_vector, vector)
            lexical_score = _lexical(
                query_text, f"{chunk.entity_name} {chunk.section} {chunk.text}"
            )
            score = max(0.0, 0.75 * ((vector_score + 1) / 2) + 0.25 * lexical_score)
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    vector_score=vector_score,
                    lexical_score=lexical_score,
                    score=score,
                )
            )
        results.sort(
            key=lambda item: (item.score, item.lexical_score, item.chunk.chunk_id), reverse=True
        )
        return results[:top_k]


class SQLiteVectorStore:
    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    fact_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    entity_kind TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    section TEXT NOT NULL,
                    text TEXT NOT NULL,
                    vector TEXT NOT NULL
                )
                """
            )

    async def upsert(self, chunks: Sequence[FactChunk], vectors: Sequence[Sequence[float]]) -> int:
        return await asyncio.to_thread(self._upsert_sync, chunks, vectors)

    def _upsert_sync(self, chunks: Sequence[FactChunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with sqlite3.connect(self.path) as connection:
            connection.executemany(
                """
                INSERT INTO chunks(
                    chunk_id, fact_id, entity_id, entity_kind, entity_name, section, text, vector
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    fact_id=excluded.fact_id, entity_id=excluded.entity_id,
                    entity_kind=excluded.entity_kind, entity_name=excluded.entity_name,
                    section=excluded.section, text=excluded.text, vector=excluded.vector
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.fact_id,
                        chunk.entity_id,
                        chunk.entity_kind,
                        chunk.entity_name,
                        chunk.section,
                        chunk.text,
                        json.dumps(list(vector)),
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
        return len(chunks)

    async def search(
        self,
        query_vector: Sequence[float],
        query_text: str,
        *,
        top_k: int,
        entity: str | None = None,
        section: str | None = None,
    ) -> list[ScoredChunk]:
        return await asyncio.to_thread(
            self._search_sync, query_vector, query_text, top_k, entity, section
        )

    def _search_sync(
        self,
        query_vector: Sequence[float],
        query_text: str,
        top_k: int,
        entity: str | None,
        section: str | None,
    ) -> list[ScoredChunk]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT fact_id, entity_id, entity_kind, entity_name, section, text, vector "
                "FROM chunks ORDER BY chunk_id",
            ).fetchall()
        results: list[ScoredChunk] = []
        for fact_id, entity_id, kind, name, chunk_section, text, raw_vector in rows:
            if entity and entity.lower() not in name.lower():
                continue
            if section and section != chunk_section:
                continue
            chunk = FactChunk(
                chunk_id=f"{fact_id}:stored",
                fact_id=fact_id,
                entity_id=entity_id,
                entity_kind=kind,
                entity_name=name,
                section=chunk_section,
                text=text,
            )
            vector = json.loads(raw_vector)
            vector_score = _cosine(query_vector, vector)
            lexical_score = _lexical(query_text, f"{name} {chunk_section} {text}")
            score = max(0.0, 0.75 * ((vector_score + 1) / 2) + 0.25 * lexical_score)
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    vector_score=vector_score,
                    lexical_score=lexical_score,
                    score=score,
                )
            )
        results.sort(
            key=lambda item: (item.score, item.lexical_score, item.chunk.fact_id), reverse=True
        )
        return results[:top_k]


class VertexVectorSearchStore:
    """Vertex AI Vector Search adapter; chunk metadata lives in Firestore."""

    def __init__(
        self,
        *,
        project: str,
        location: str,
        index_name: str,
        endpoint_name: str,
        deployed_index_id: str,
        metadata_collection: str,
    ) -> None:
        from google.cloud import aiplatform, firestore
        from google.cloud.aiplatform_v1.types import IndexDatapoint

        self._index_datapoint = IndexDatapoint
        aiplatform.init(project=project, location=location)
        self._index = aiplatform.MatchingEngineIndex(index_name=index_name)
        self._endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_name)
        self._deployed_index_id = deployed_index_id
        self._firestore = firestore.Client(project=project)
        self._collection = metadata_collection

    async def upsert(self, chunks: Sequence[FactChunk], vectors: Sequence[Sequence[float]]) -> int:
        return await asyncio.to_thread(self._upsert_sync, chunks, vectors)

    def _upsert_sync(self, chunks: Sequence[FactChunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        datapoints = []
        batch = self._firestore.batch()
        for chunk, vector in zip(chunks, vectors, strict=True):
            datapoints.append(
                self._index_datapoint(
                    datapoint_id=chunk.chunk_id,
                    feature_vector=list(vector),
                    restricts=[
                        {"namespace": "entity", "allow_list": [chunk.entity_name]},
                        {"namespace": "section", "allow_list": [chunk.section or "__empty__"]},
                    ],
                )
            )
            batch.set(
                self._firestore.collection(self._collection).document(chunk.chunk_id),
                chunk.model_dump(),
            )
        if datapoints:
            self._index.upsert_datapoints(datapoints)
            batch.commit()
        return len(chunks)

    async def search(
        self,
        query_vector: Sequence[float],
        query_text: str,
        *,
        top_k: int,
        entity: str | None = None,
        section: str | None = None,
    ) -> list[ScoredChunk]:
        del query_text
        return await asyncio.to_thread(self._search_sync, query_vector, top_k, entity, section)

    def _search_sync(
        self,
        query_vector: Sequence[float],
        top_k: int,
        entity: str | None,
        section: str | None,
    ) -> list[ScoredChunk]:
        from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace

        filters = []
        if entity:
            filters.append(Namespace(name="entity", allow_tokens=[entity], deny_tokens=[]))
        if section:
            filters.append(Namespace(name="section", allow_tokens=[section], deny_tokens=[]))
        neighbors = self._endpoint.find_neighbors(
            deployed_index_id=self._deployed_index_id,
            queries=[list(query_vector)],
            num_neighbors=top_k,
            filter=filters or None,
        )[0]
        results: list[ScoredChunk] = []
        for neighbor in neighbors:
            document = self._firestore.collection(self._collection).document(neighbor.id).get()
            if not document.exists:
                continue
            chunk = FactChunk.model_validate(document.to_dict())
            distance = float(getattr(neighbor, "distance", 0.0) or 0.0)
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    vector_score=max(-1.0, min(1.0, 1.0 - distance)),
                    lexical_score=0.0,
                    score=max(0.0, 1.0 - distance),
                )
            )
        return results
