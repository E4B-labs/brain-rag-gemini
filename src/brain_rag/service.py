import asyncio
import logging
import time
import uuid

from .chunking import chunk_facts
from .config import Settings
from .embeddings import EmbeddingResult
from .guardrails import validate_grounding
from .models import (
    FactChunk,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ScoredChunk,
    SourceCitation,
)
from .ports import BrainSource, Embedder, Generator, VectorStore
from .retrieval import rerank

logger = logging.getLogger("brain_rag.service")


class RagService:
    def __init__(
        self,
        *,
        settings: Settings,
        source: BrainSource,
        embedder: Embedder,
        store: VectorStore,
        generator: Generator,
    ) -> None:
        self.settings = settings
        self.source = source
        self.embedder = embedder
        self.store = store
        self.generator = generator

    async def ingest(self, workspace_id: str | None = None) -> IngestResponse:
        workspace = workspace_id or self.settings.brain_workspace_id
        facts = await self.source.fetch_facts(workspace)
        chunks = chunk_facts(facts)
        total_written = 0
        batch_size = 128
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embedding = await self.embedder.embed(
                [chunk.text for chunk in batch], task_type="RETRIEVAL_DOCUMENT"
            )
            total_written += await self.store.upsert(batch, embedding.vectors)
        return IngestResponse(
            facts_seen=len(facts),
            chunks_written=total_written,
            embedding_model=getattr(self.settings, "embedding_model", "unknown"),
            vector_store=type(self.store).__name__,
        )

    async def query(self, request: QueryRequest) -> QueryResponse:
        if len(request.question) > self.settings.max_query_chars:
            raise ValueError("Question exceeds the configured character limit")
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        embedding, contexts, retrieval_stats = await self._retrieve(request, request_id)
        if not contexts:
            raise ValueError("No grounded facts found for this query")
        generation_contexts = contexts[: self.settings.rag_generation_contexts]
        generation = await self.generator.generate(
            request.question,
            generation_contexts,
            max_output_tokens=self.settings.max_output_tokens,
        )
        payload = generation.payload
        validate_grounding(payload, generation_contexts)
        total_cost = embedding.estimated_cost_usd + generation.estimated_cost_usd
        if total_cost > self.settings.max_query_cost_usd:
            raise ValueError("Query cost limit exceeded")
        by_id = {context.fact_id: context for context in contexts}
        citations = [
            SourceCitation(
                fact_id=fact_id,
                entity=by_id[fact_id].entity_name,
                section=by_id[fact_id].section,
                quote=by_id[fact_id].text,
            )
            for fact_id in payload.citations
        ]
        response = QueryResponse(
            answer=payload.answer,
            citations=citations,
            model=generation.model,
            retrieved_count=len(contexts),
            latency_ms=(time.perf_counter() - started) * 1000,
            estimated_cost_usd=total_cost,
            request_id=request_id,
        )
        logger.info(
            "query_breakdown request_id=%s embedding_ms=%.2f retrieval_ms=%.2f "
            "generation_ms=%.2f non_model_ms=%.2f",
            request_id,
            retrieval_stats["embedding_ms"],
            retrieval_stats["retrieval_ms"],
            generation.latency_ms,
            max(0.0, response.latency_ms - generation.latency_ms),
        )
        logger.info(
            "query_complete request_id=%s model=%s retrieved_count=%d latency_ms=%.2f "
            "estimated_cost_usd=%.8f",
            request_id,
            response.model,
            response.retrieved_count,
            response.latency_ms,
            response.estimated_cost_usd,
        )
        return response

    async def retrieve(self, request: QueryRequest) -> tuple[EmbeddingResult, list[FactChunk]]:
        embedding, contexts, _ = await self._retrieve(request, None)
        return embedding, contexts

    async def _retrieve(
        self, request: QueryRequest, request_id: str | None
    ) -> tuple[EmbeddingResult, list[FactChunk], dict[str, float]]:
        started = time.perf_counter()

        async def embed_query() -> tuple[EmbeddingResult, float]:
            phase_started = time.perf_counter()
            result = await self.embedder.embed(
                [request.question], task_type="RETRIEVAL_QUERY"
            )
            return result, (time.perf_counter() - phase_started) * 1000

        async def retrieve_text() -> tuple[list[ScoredChunk], float]:
            phase_started = time.perf_counter()
            result = await self.store.search(
                [],
                request.question,
                top_k=min(20, request.top_k * self.settings.rag_candidate_multiplier),
                entity=request.entity,
                section=request.section,
            )
            return result, (time.perf_counter() - phase_started) * 1000

        if getattr(self.store, "supports_text_retrieval", False):
            (embedding, embedding_ms), (candidates, retrieval_ms) = await asyncio.gather(
                embed_query(), retrieve_text()
            )
        else:
            embedding, embedding_ms = await embed_query()
            retrieval_started = time.perf_counter()
            candidates = await self.store.search(
                embedding.vectors[0],
                request.question,
                top_k=min(20, request.top_k * self.settings.rag_candidate_multiplier),
                entity=request.entity,
                section=request.section,
            )
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        ranked = rerank(
            candidates, entity=request.entity, section=request.section, top_k=request.top_k
        )
        contexts = [item.chunk for item in ranked]
        stats: dict[str, float] = {
            "embedding_ms": embedding_ms,
            "retrieval_ms": retrieval_ms,
            "total_ms": (time.perf_counter() - started) * 1000,
        }
        if request_id:
            logger.info(
                "retrieval_breakdown request_id=%s embedding_ms=%.2f retrieval_ms=%.2f "
                "contexts=%d",
                request_id,
                stats["embedding_ms"],
                stats["retrieval_ms"],
                len(contexts),
            )
        return embedding, contexts, stats
