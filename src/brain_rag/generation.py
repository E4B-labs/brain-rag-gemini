import asyncio
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .config import Settings
from .models import AnswerPayload, FactChunk
from .observability import timed

logger = logging.getLogger("brain_rag.generation")

SYSTEM_PROMPT = """
You answer questions about the TaskTree Brain. Use only the supplied facts.
Every factual claim must be supported by one or more citation IDs from the facts.
Never invent a citation ID. If the facts do not answer the question, say so clearly
and cite the closest relevant fact only when it supports that limitation.
Return JSON matching the requested schema.
""".strip()


@dataclass(frozen=True)
class GenerationResult:
    payload: AnswerPayload
    model: str
    latency_ms: float
    estimated_cost_usd: float


def needs_fallback(question: str) -> bool:
    words = re.findall(r"\w+", question.lower())
    difficult_markers = {
        "dlaczego",
        "porownaj",
        "porównaj",
        "konsekwencje",
        "tradeoff",
        "strategia",
        "plan",
        "analizuj",
        "zaleznosc",
        "zależność",
    }
    return len(words) > 35 or len(difficult_markers.intersection(words)) >= 2


def build_prompt(question: str, contexts: Sequence[FactChunk]) -> str:
    facts = "\n\n".join(
        f"FACT_ID: {context.fact_id}\n"
        f"ENTITY: {context.entity_name} ({context.entity_kind})\n"
        f"SECTION: {context.section or 'unclassified'}\n"
        f"TEXT: {context.text}"
        for context in contexts
    )
    return f"Question: {question}\n\nSource facts:\n{facts}"


class VertexGeminiGenerator:
    def __init__(self, settings: Settings) -> None:
        from google import genai

        if not settings.google_cloud_project:
            raise ValueError("GOOGLE_CLOUD_PROJECT is required for Vertex AI generation")
        self.settings = settings
        self.client = genai.Client(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )
        self._cache_name: str | None = None

    async def generate(
        self,
        question: str,
        contexts: Sequence[FactChunk],
        *,
        max_output_tokens: int,
    ) -> GenerationResult:
        model = (
            self.settings.fallback_model
            if needs_fallback(question)
            else self.settings.generation_model
        )
        finish = timed("generation")
        cache_name = await self._ensure_context_cache(model)

        def call() -> object:
            from google.genai.types import GenerateContentConfig

            config = GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnswerPayload.model_json_schema(),
                max_output_tokens=max_output_tokens,
                temperature=0.1,
                cached_content=cache_name,
                system_instruction=None if cache_name else SYSTEM_PROMPT,
            )
            return self.client.models.generate_content(
                model=model,
                contents=build_prompt(question, contexts),
                config=config,
            )

        response = await asyncio.to_thread(call)
        try:
            parsed = getattr(response, "parsed", None)
            if parsed is not None:
                payload = AnswerPayload.model_validate(parsed)
            else:
                text = str(getattr(response, "text", "") or "").strip()
                if not text:
                    raise RuntimeError("Vertex AI returned an empty response")
                payload = AnswerPayload.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError("Vertex AI response was not valid grounded JSON") from exc

        metadata = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(metadata, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(metadata, "candidates_token_count", 0) or 0)
        cost = self._estimate_cost(model, input_tokens, output_tokens)
        usage = finish(
            model=model,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return GenerationResult(payload, model, usage.latency_ms, usage.estimated_cost_usd)

    async def _ensure_context_cache(self, model: str) -> str | None:
        if not self.settings.enable_context_cache or self._cache_name:
            return self._cache_name
        try:

            def create() -> object:
                from google.genai.types import Content, CreateCachedContentConfig, Part

                return self.client.caches.create(
                    model=model,
                    config=CreateCachedContentConfig(
                        contents=[
                            Content(role="user", parts=[Part.from_text(text="Grounded RAG policy")])
                        ],
                        system_instruction=SYSTEM_PROMPT,
                        display_name="brain-rag-system-prompt",
                        ttl="3600s",
                    ),
                )

            cache = await asyncio.to_thread(create)
            self._cache_name = str(getattr(cache, "name", "") or "") or None
        except (
            Exception
        ) as exc:  # Context cache is an optimization, never a correctness dependency.
            logger.warning("context_cache_unavailable error=%s", exc)
        return self._cache_name

    @staticmethod
    def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        # Conservative defaults for observability; update with the project's pricing sheet.
        input_rate = 0.50 if "flash" in model else 2.00
        output_rate = 3.00 if "flash" in model else 12.00
        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


class FakeGenerator:
    def __init__(self, payload: AnswerPayload, model: str = "fake-gemini") -> None:
        self.payload = payload
        self.model = model
        self.calls: list[tuple[str, list[str]]] = []

    async def generate(
        self,
        question: str,
        contexts: Sequence[FactChunk],
        *,
        max_output_tokens: int,
    ) -> GenerationResult:
        del max_output_tokens
        self.calls.append((question, [context.fact_id for context in contexts]))
        return GenerationResult(self.payload, self.model, 0.1, 0.0)


class LocalGroundedGenerator:
    """Offline generator used only when no Vertex project is configured."""

    async def generate(
        self,
        question: str,
        contexts: Sequence[FactChunk],
        *,
        max_output_tokens: int,
    ) -> GenerationResult:
        del max_output_tokens
        first = contexts[0]
        payload = AnswerPayload(
            answer=(
                f"Tryb lokalny: najbardziej pasujacy fakt dla pytania '{question}' to: {first.text}"
            ),
            citations=[first.fact_id],
            confidence="low",
        )
        return GenerationResult(payload, "local-demo", 0.0, 0.0)
