from brain_rag.chunking import chunk_fact, chunk_facts


def test_chunk_fact_preserves_fact_id_and_metadata(facts) -> None:
    chunks = chunk_fact(facts[0], max_chars=18, overlap=3)
    assert len(chunks) > 1
    assert {chunk.fact_id for chunk in chunks} == {"f-1"}
    assert {chunk.entity_name for chunk in chunks} == {"TaskTree"}


def test_alias_observation_is_not_indexed(facts) -> None:
    alias = facts[0].model_copy(update={"body": "Szukaj takze po: tasktree, tt."})
    assert chunk_fact(alias) == []


def test_chunk_facts_skips_aliases(facts) -> None:
    alias = facts[0].model_copy(update={"body": "Szukaj takze po: tasktree."})
    chunks = chunk_facts([facts[0], alias])
    assert {chunk.fact_id for chunk in chunks} == {"f-1"}
