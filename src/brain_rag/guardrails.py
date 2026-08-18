from collections.abc import Sequence

from .models import AnswerPayload, FactChunk


class GroundingError(ValueError):
    pass


def validate_grounding(payload: AnswerPayload, contexts: Sequence[FactChunk]) -> None:
    available = {context.fact_id for context in contexts}
    invalid = sorted(set(payload.citations) - available)
    if invalid:
        raise GroundingError(f"Unknown citation IDs: {', '.join(invalid)}")
    if not payload.citations:
        raise GroundingError("Grounded answers must contain at least one citation")
