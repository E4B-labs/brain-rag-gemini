from collections.abc import Iterable

from .models import BrainFact, FactChunk

ALIAS_PREFIX = "Szukaj takze po:"


def chunk_fact(fact: BrainFact, *, max_chars: int = 1200, overlap: int = 120) -> list[FactChunk]:
    if max_chars <= overlap:
        raise ValueError("max_chars must be greater than overlap")
    if fact.body.startswith(ALIAS_PREFIX):
        return []
    text = fact.body.strip()
    chunks: list[FactChunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("; ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                FactChunk(
                    chunk_id=f"{fact.fact_id}:{index}",
                    fact_id=fact.fact_id,
                    entity_id=fact.entity_id,
                    entity_kind=fact.entity_kind,
                    entity_name=fact.entity_name,
                    section=fact.section,
                    text=piece,
                )
            )
            index += 1
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_facts(facts: Iterable[BrainFact], **kwargs: int) -> list[FactChunk]:
    chunks: list[FactChunk] = []
    for fact in facts:
        chunks.extend(chunk_fact(fact, **kwargs))
    return chunks
