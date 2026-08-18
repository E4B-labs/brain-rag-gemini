import asyncio
import importlib
import json
import math
import re
import sqlite3
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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


class RagEngineStore:
    """Vertex AI RAG Engine adapter using a managed RAG corpus.

    RAG Engine owns the vector database; no user-managed Vector Search index or
    endpoint is created by this adapter. Fact metadata is stored in the file
    payload so retrieved contexts remain citation-safe across process restarts.
    """

    def __init__(
        self,
        *,
        project: str,
        location: str,
        corpus_name: str,
        staging_bucket: str | None = None,
    ) -> None:
        import importlib

        vertexai: Any = importlib.import_module("vertexai")
        self._rag: Any = importlib.import_module("vertexai.rag")
        vertexai.init(project=project, location=location)
        self._corpus_name = corpus_name
        self._staging_bucket = staging_bucket
        self._chunks: dict[str, FactChunk] = {}

    async def upsert(self, chunks: Sequence[FactChunk], vectors: Sequence[Sequence[float]]) -> int:
        del vectors
        return await asyncio.to_thread(self._upsert_sync, chunks)

    def _upsert_sync(self, chunks: Sequence[FactChunk]) -> int:
        paths: list[str] = []
        for chunk in chunks:
            metadata = json.dumps(
                {
                    "chunk_id": chunk.chunk_id,
                    "fact_id": chunk.fact_id,
                    "entity_id": chunk.entity_id,
                    "entity_kind": chunk.entity_kind,
                    "entity_name": chunk.entity_name,
                    "section": chunk.section,
                },
                ensure_ascii=True,
            )
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".txt", delete=False
            ) as file:
                file.write(f"{metadata}\n{chunk.text}\n")
                local_path = file.name
            paths.append(local_path)
            self._chunks[chunk.chunk_id] = chunk
        try:
            if paths:
                if self._staging_bucket:
                    self._import_from_gcs(paths, chunks)
                else:
                    self._upload_locally(paths, chunks)
        finally:
            for path in paths:
                for _ in range(5):
                    try:
                        Path(path).unlink(missing_ok=True)
                        break
                    except PermissionError:
                        time.sleep(1)
        return len(chunks)

    def _upload_locally(self, paths: list[str], chunks: Sequence[FactChunk]) -> None:
        for path, chunk in zip(paths, chunks, strict=True):
            for attempt in range(6):
                try:
                    self._rag.upload_file(
                        corpus_name=self._corpus_name,
                        path=path,
                        display_name=chunk.chunk_id,
                    )
                    break
                except RuntimeError as exc:
                    if "429" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc):
                        raise
                    if attempt == 5:
                        raise
                    time.sleep(10 * (attempt + 1))

    def _import_from_gcs(self, paths: list[str], chunks: Sequence[FactChunk]) -> None:
        from concurrent.futures import ThreadPoolExecutor

        storage: Any = importlib.import_module("google.cloud.storage")
        client = storage.Client()
        bucket = client.bucket(self._staging_bucket)
        names = [f"brain-rag/{chunk.chunk_id.replace(':', '_')}.txt" for chunk in chunks]

        def upload(item: tuple[str, str]) -> None:
            path, name = item
            bucket.blob(name).upload_from_filename(path, content_type="text/plain")

        with ThreadPoolExecutor(max_workers=min(16, len(paths))) as executor:
            list(executor.map(upload, zip(paths, names, strict=True)))
        uris = [f"gs://{self._staging_bucket}/{name}" for name in names]
        operations = []
        for start in range(0, len(uris), 25):
            operations.append(self._rag.import_files_async(
                corpus_name=self._corpus_name,
                paths=uris[start : start + 25],
            ))
        for operation in operations:
            operation.result()
        for name in names:
            bucket.blob(name).delete()

    async def search(
        self,
        query_vector: Sequence[float],
        query_text: str,
        *,
        top_k: int,
        entity: str | None = None,
        section: str | None = None,
    ) -> list[ScoredChunk]:
        del query_vector
        return await asyncio.to_thread(
            self._search_sync, query_text, top_k, entity, section
        )

    def _search_sync(
        self,
        query_text: str,
        top_k: int,
        entity: str | None,
        section: str | None,
    ) -> list[ScoredChunk]:
        response = self._rag.retrieval_query(
            text=query_text,
            rag_resources=[self._rag.RagResource(rag_corpus=self._corpus_name)],
            rag_retrieval_config=self._rag.RagRetrievalConfig(top_k=max(top_k * 4, top_k)),
        )
        results: list[ScoredChunk] = []
        for context in response.contexts.contexts:
            chunk = self._decode_context(context)
            if chunk is None:
                continue
            if entity and entity.lower() not in chunk.entity_name.lower():
                continue
            if section and section != chunk.section:
                continue
            raw_score = float(getattr(context, "score", 0.0) or 0.0)
            vector_score = 1.0 - raw_score if raw_score > 1.0 else raw_score
            vector_score = max(-1.0, min(1.0, vector_score))
            lexical_score = _lexical(
                query_text, f"{chunk.entity_name} {chunk.section} {chunk.text}"
            )
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    vector_score=vector_score,
                    lexical_score=lexical_score,
                    score=max(0.0, 0.75 * max(vector_score, 0.0) + 0.25 * lexical_score),
                )
            )
        results.sort(key=lambda item: (item.score, item.lexical_score), reverse=True)
        return results[:top_k]

    def _decode_context(self, context: object) -> FactChunk | None:
        display_name = str(getattr(context, "source_display_name", "") or "")
        if display_name in self._chunks:
            return self._chunks[display_name]
        raw_text = str(getattr(context, "text", "") or "")
        metadata_line, _, body = raw_text.partition("\n")
        try:
            metadata = json.loads(metadata_line)
            return FactChunk(
                chunk_id=metadata["chunk_id"],
                fact_id=metadata["fact_id"],
                entity_id=metadata["entity_id"],
                entity_kind=metadata["entity_kind"],
                entity_name=metadata["entity_name"],
                section=metadata["section"],
                text=body.strip(),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
