import pytest
from pydantic import ValidationError

from brain_rag.models import AnswerPayload, BrainFact, QueryRequest


def test_brain_fact_rejects_empty_body() -> None:
    with pytest.raises(ValidationError):
        BrainFact(
            fact_id="x", entity_id="e", entity_kind="note", entity_name="x", body=" "
        )


def test_query_request_has_bounded_top_k() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="hello", top_k=21)


def test_answer_payload_requires_citations() -> None:
    with pytest.raises(ValidationError):
        AnswerPayload(answer="answer", citations=[])

