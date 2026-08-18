import json
from types import SimpleNamespace

import pytest

from brain_rag.chunking import chunk_facts
from brain_rag.stores import InMemoryVectorStore, RagEngineStore, SQLiteVectorStore


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


def test_rag_engine_store_decodes_citation_metadata(facts) -> None:
    chunk = chunk_facts(facts)[0]
    store = object.__new__(RagEngineStore)
    store._chunks = {}
    metadata = json.dumps(
        {
            "chunk_id": chunk.chunk_id,
            "fact_id": chunk.fact_id,
            "entity_id": chunk.entity_id,
            "entity_kind": chunk.entity_kind,
            "entity_name": chunk.entity_name,
            "section": chunk.section,
        }
    )
    context = SimpleNamespace(text=f"{metadata}\n{chunk.text}", source_display_name="")

    decoded = store._decode_context(context)

    assert decoded is not None
    assert decoded.fact_id == chunk.fact_id
    assert decoded.text == chunk.text


@pytest.mark.asyncio
async def test_rag_engine_store_retrieval_applies_metadata_filters(facts) -> None:
    chunk = chunk_facts(facts)[0]
    metadata = json.dumps(
        {
            "chunk_id": chunk.chunk_id,
            "fact_id": chunk.fact_id,
            "entity_id": chunk.entity_id,
            "entity_kind": chunk.entity_kind,
            "entity_name": chunk.entity_name,
            "section": chunk.section,
        }
    )

    class FakeRag:
        @staticmethod
        def RagResource(**kwargs):
            return kwargs

        @staticmethod
        def RagRetrievalConfig(**kwargs):
            return kwargs

        def retrieval_query(self, **kwargs):
            assert kwargs["rag_resources"] == [{"rag_corpus": "corpus"}]
            return SimpleNamespace(
                contexts=SimpleNamespace(
                    contexts=[
                        SimpleNamespace(
                            text=f"{metadata}\n{chunk.text}",
                            source_display_name="",
                            score=0.9,
                        )
                    ]
                )
            )

    store = object.__new__(RagEngineStore)
    store._rag = FakeRag()
    store._corpus_name = "corpus"
    store._chunks = {}

    results = await store.search([], "Firestore", top_k=3, entity="Brain", section="stack")

    assert results == []
