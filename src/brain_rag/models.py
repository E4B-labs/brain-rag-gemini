from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BrainFact(BaseModel):
    model_config = ConfigDict(frozen=True)

    fact_id: str
    entity_id: str
    entity_kind: str
    entity_name: str
    body: str = Field(min_length=1)
    section: str = ""
    source: str = "mcp"
    source_ref: str | None = None
    occurred_at: datetime | None = None

    @field_validator("body", "entity_name")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be empty")
        return value


class FactChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    fact_id: str
    entity_id: str
    entity_kind: str
    entity_name: str
    section: str
    text: str = Field(min_length=1)


class ScoredChunk(BaseModel):
    chunk: FactChunk
    vector_score: float = Field(ge=-1, le=1)
    lexical_score: float = Field(ge=0, le=1)
    score: float = Field(ge=0)


class AnswerPayload(BaseModel):
    answer: str = Field(min_length=1, max_length=12_000)
    citations: list[str] = Field(min_length=1, max_length=20)
    confidence: Literal["high", "medium", "low"] = "medium"


class SourceCitation(BaseModel):
    fact_id: str
    entity: str
    section: str
    quote: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=20_000)
    top_k: int = Field(default=5, ge=1, le=20)
    entity: str | None = Field(default=None, max_length=200)
    section: str | None = Field(default=None, max_length=50)


class QueryResponse(BaseModel):
    answer: str
    citations: list[SourceCitation]
    model: str
    retrieved_count: int
    latency_ms: float = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    request_id: str


class IngestResponse(BaseModel):
    facts_seen: int
    chunks_written: int
    embedding_model: str
    vector_store: str
