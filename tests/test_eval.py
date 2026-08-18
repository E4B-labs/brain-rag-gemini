from brain_rag.eval import faithfulness, recall_at_k
from brain_rag.models import FactChunk


def test_recall_at_k() -> None:
    chunks = [
        FactChunk(
            chunk_id="a:0",
            fact_id="a",
            entity_id="e",
            entity_kind="note",
            entity_name="A",
            section="",
            text="a",
        ),
        FactChunk(
            chunk_id="b:0",
            fact_id="b",
            entity_id="e",
            entity_kind="note",
            entity_name="B",
            section="",
            text="b",
        ),
    ]
    assert recall_at_k(chunks, frozenset({"a"}), 1) == 1.0


def test_faithfulness_requires_existing_citations() -> None:
    assert faithfulness("answer", ["a"], frozenset({"a"})) == 1.0
    assert faithfulness("answer", ["x"], frozenset({"a"})) == 0.0
