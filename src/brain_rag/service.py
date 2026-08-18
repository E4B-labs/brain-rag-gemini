import time
import uuid

from .chunking import chunk_facts
from .config import Settings
from .guardrails import validate_grounding
from .models import IngestResponse, QueryRequest, QueryResponse, SourceCitation
from .ports import BrainSource, Embedder, Generator, VectorStore
from .retrieval import rerank


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
        batch_size = 16
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
        embedding = await self.embedder.embed([request.question], task_type="RETRIEVAL_QUERY")
        candidates = await self.store.search(
            embedding.vectors[0],
            request.question,
            top_k=min(50, request.top_k * 4),
            entity=request.entity,
            section=request.section,
        )
        ranked = rerank(
            candidates, entity=request.entity, section=request.section, top_k=request.top_k
        )
        contexts = [item.chunk for item in ranked]
        if not contexts:
            raise ValueError("No grounded facts found for this query")
        generation = await self.generator.generate(
            request.question,
            contexts,
            max_output_tokens=self.settings.max_output_tokens,
        )
        payload = generation.payload
        validate_grounding(payload, contexts)
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
        return QueryResponse(
            answer=payload.answer,
            citations=citations,
            model=generation.model,
            retrieved_count=len(contexts),
            latency_ms=(time.perf_counter() - started) * 1000,
            estimated_cost_usd=total_cost,
            request_id=str(uuid.uuid4()),
        )
