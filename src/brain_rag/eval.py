import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from .config import Settings
from .models import FactChunk
from .observability import timed


@dataclass(frozen=True)
class GoldCase:
    question: str
    relevant_fact_ids: frozenset[str]
    expected_answer: str


def recall_at_k(retrieved: Sequence[FactChunk], relevant_fact_ids: frozenset[str], k: int) -> float:
    if not relevant_fact_ids:
        return 1.0
    found = {chunk.fact_id for chunk in retrieved[:k]}
    return len(found & relevant_fact_ids) / len(relevant_fact_ids)


def faithfulness(
    answer: str, citations: Sequence[str], available_fact_ids: frozenset[str]
) -> float:
    del answer
    return 1.0 if citations and set(citations) <= available_fact_ids else 0.0


class JudgePayload(BaseModel):
    faithful: bool
    rationale: str = Field(min_length=1, max_length=2_000)


class VertexFaithfulnessJudge:
    """LLM-as-judge using the configured Flash-Lite model."""

    def __init__(self, settings: Settings) -> None:
        from google import genai

        if not settings.google_cloud_project:
            raise ValueError("GOOGLE_CLOUD_PROJECT is required for the Vertex judge")
        self.settings = settings
        self.client = genai.Client(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )

    async def judge(
        self,
        question: str,
        answer: str,
        citations: Sequence[FactChunk],
    ) -> float:
        prompt = (
            "Judge whether the answer is fully supported by the cited facts. "
            "Return faithful=true only when every material claim is supported.\n\n"
            f"Question: {question}\nAnswer: {answer}\nFacts:\n"
            + "\n".join(f"[{fact.fact_id}] {fact.text}" for fact in citations)
        )
        finish = timed("faithfulness_judge")

        def call() -> Any:
            from google.genai.types import GenerateContentConfig

            return self.client.models.generate_content(
                model=self.settings.judge_model,
                contents=prompt,
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=JudgePayload.model_json_schema(),
                    max_output_tokens=200,
                    temperature=0,
                ),
            )

        response = await asyncio.to_thread(call)
        payload = JudgePayload.model_validate(json.loads(str(response.text)))
        metadata = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(metadata, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(metadata, "candidates_token_count", 0) or 0)
        finish(
            model=self.settings.judge_model,
            cost=(input_tokens * 0.10 + output_tokens * 0.40) / 1_000_000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return 1.0 if payload.faithful else 0.0
