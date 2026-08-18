import pytest
from conftest import FakeEmbedder, FakeGenerator

from brain_rag.brain import StaticBrainSource
from brain_rag.config import Settings
from brain_rag.generation import needs_fallback
from brain_rag.guardrails import GroundingError, validate_grounding
from brain_rag.models import AnswerPayload, QueryRequest
from brain_rag.service import RagService
from brain_rag.stores import InMemoryVectorStore


def make_service(facts, tmp_path, generator=None):
    embedder = FakeEmbedder()
    generator = generator or FakeGenerator()
    service = RagService(
        settings=Settings(sqlite_path=str(tmp_path / "vectors.sqlite3"), max_output_tokens=120),
        source=StaticBrainSource(facts),
        embedder=embedder,
        store=InMemoryVectorStore(),
        generator=generator,
    )
    return service, embedder, generator


@pytest.mark.asyncio
async def test_ingest_embeds_and_writes_all_facts(facts, tmp_path) -> None:
    service, embedder, _ = make_service(facts, tmp_path)
    result = await service.ingest("workspace")
    assert result.facts_seen == 3
    assert result.chunks_written == 3
    assert embedder.calls[0][1] == "RETRIEVAL_DOCUMENT"


@pytest.mark.asyncio
async def test_query_returns_only_grounded_citations(facts, tmp_path) -> None:
    service, _, generator = make_service(facts, tmp_path)
    await service.ingest("workspace")
    response = await service.query(QueryRequest(question="Where is TaskTree data?", top_k=2))
    assert response.citations[0].fact_id == "f-1"
    assert generator.calls[0][1]


@pytest.mark.asyncio
async def test_query_rejects_unknown_citation(facts, tmp_path) -> None:
    service, _, _ = make_service(
        facts, tmp_path, generator=FakeGenerator(citation_id="not-retrieved")
    )
    await service.ingest("workspace")
    with pytest.raises(GroundingError):
        await service.query(QueryRequest(question="Where is TaskTree data?"))


@pytest.mark.asyncio
async def test_query_enforces_cost_limit(facts, tmp_path) -> None:
    generator = FakeGenerator(cost=1.0)
    service, _, _ = make_service(facts, tmp_path, generator=generator)
    service.settings.max_query_cost_usd = 0.01
    await service.ingest("workspace")
    with pytest.raises(ValueError, match="cost limit"):
        await service.query(QueryRequest(question="Where is TaskTree data?"))


def test_difficult_questions_route_to_fallback() -> None:
    assert needs_fallback("Porownaj strategie wdrozenia i wyjasnij konsekwencje decyzji")
    assert not needs_fallback("Where is Brain?")


def test_guardrail_accepts_existing_ids(facts) -> None:
    validate_grounding(AnswerPayload(answer="ok", citations=["f-1"]), facts)
