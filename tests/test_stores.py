import pytest

from brain_rag.chunking import chunk_facts
from brain_rag.stores import InMemoryVectorStore, SQLiteVectorStore


@pytest.mark.asyncio
async def test_in_memory_store_combines_vector_and_lexical_scores(facts) -> None:
    chunks = chunk_facts(facts)
    store = InMemoryVectorStore()
    await store.upsert(chunks, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    results = await store.search([0, 1, 0], "Supabase observations", top_k=2)
    assert results[0].chunk.fact_id == "f-2"
    assert results[0].lexical_score > 0


@pytest.mark.asyncio
async def test_in_memory_store_filters_entity_and_section(facts) -> None:
    chunks = chunk_facts(facts)
    store = InMemoryVectorStore()
    await store.upsert(chunks, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    results = await store.search([1, 0, 0], "anything", top_k=5, entity="Brain", section="stan")
    assert [item.chunk.fact_id for item in results] == ["f-3"]


@pytest.mark.asyncio
async def test_sqlite_store_round_trip(tmp_path, facts) -> None:
    store = SQLiteVectorStore(str(tmp_path / "vectors.sqlite3"))
    chunks = chunk_facts(facts)
    await store.upsert(chunks, [[1, 0, 0, 0]] * len(chunks))
    results = await store.search([1, 0, 0, 0], "Firestore", top_k=2)
    assert results[0].chunk.fact_id == "f-1"
