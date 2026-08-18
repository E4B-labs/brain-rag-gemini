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

        def call(
            output_tokens: int,
            concise: bool = False,
            use_cache: bool = True,
            structured: bool = True,
        ) -> object:
            from google.genai.types import GenerateContentConfig, ThinkingConfig, ThinkingLevel

            prompt = build_prompt(question, contexts)
            if concise:
                prompt += (
                    "\nReturn only valid JSON. Keep the answer under 60 words and cite "
                    "one or more supplied FACT_ID values."
                )
            thinking_level = self.settings.thinking_level.upper()
            if "pro" in model.lower() and thinking_level == "MINIMAL":
                thinking_level = "LOW"
            thinking_config = ThinkingConfig(
                thinking_level=ThinkingLevel(thinking_level)
            )
            if structured:
                config = GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AnswerPayload.model_json_schema(),
                    max_output_tokens=output_tokens,
                    temperature=0 if concise else 0.1,
                    cached_content=cache_name if use_cache else None,
                    system_instruction=None if cache_name and use_cache else SYSTEM_PROMPT,
                    thinking_config=thinking_config,
                )
            else:
                config = GenerateContentConfig(
                    max_output_tokens=output_tokens,
                    temperature=0 if concise else 0.1,
                    cached_content=None,
                    system_instruction=SYSTEM_PROMPT,
                    thinking_config=thinking_config,
                )
            return self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

        response = await asyncio.to_thread(call, max_output_tokens)

        def parse_response(candidate: object) -> AnswerPayload:
            parsed = getattr(candidate, "parsed", None)
            if parsed is not None:
                payload = AnswerPayload.model_validate(parsed)
            else:
                text = str(getattr(candidate, "text", "") or "").strip()
                if not text:
                    raise RuntimeError("Vertex AI returned an empty response")
                payload = AnswerPayload.model_validate(json.loads(text))
            return payload

        try:
            payload = parse_response(response)
        except (RuntimeError, json.JSONDecodeError, ValueError):
            logger.warning("generation_json_retry model=%s", model)
            response = await asyncio.to_thread(call, min(max_output_tokens, 256), True, False)
            try:
                payload = parse_response(response)
            except (RuntimeError, json.JSONDecodeError, ValueError) as retry_exc:
                logger.warning("generation_text_fallback model=%s", model)
                response = await asyncio.to_thread(
                    call, min(max_output_tokens, 256), True, False, False
                )
                text = str(getattr(response, "text", "") or "").strip()
                if text:
                    payload = AnswerPayload(
                        answer=text,
                        citations=[contexts[0].fact_id],
                        confidence="low",
                    )
                elif contexts:
                    payload = AnswerPayload(
                        answer=contexts[0].text,
                        citations=[contexts[0].fact_id],
                        confidence="low",
                    )
                else:
                    raise RuntimeError(
                        "Vertex AI response was not valid grounded JSON"
                    ) from retry_exc

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
        # Vertex requires at least 1,024 cached input tokens; this system prompt is
        # intentionally short, so avoid a guaranteed failing cache API round trip.
        if not self.settings.enable_context_cache or len(SYSTEM_PROMPT.split()) < 700:
            return None
        if self._cache_name:
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
