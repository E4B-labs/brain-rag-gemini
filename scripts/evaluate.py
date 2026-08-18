import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from brain_rag.api import build_service
from brain_rag.config import get_settings
from brain_rag.eval import VertexFaithfulnessJudge, faithfulness, recall_at_k
from brain_rag.guardrails import validate_grounding
from brain_rag.models import QueryRequest


def build_real_brain_cases(facts: list[Any], limit: int = 10) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, fact in enumerate(facts[:limit], start=1):
        cases.append(
            {
                "id": f"brain-{index}",
                "question": (
                    f"Jaki fakt Brain zapisuje o encji {fact.entity_name!r} "
                    f"w sekcji {fact.section!r}?"
                ),
                "relevant_fact_ids": [fact.fact_id],
            }
        )
    return cases


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="eval/golden.json")
    parser.add_argument("--real-brain", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    settings = get_settings()
    service = build_service()
    judge = VertexFaithfulnessJudge(settings) if settings.google_cloud_project else None
    if args.real_brain:
        facts = await service.source.fetch_facts(settings.brain_workspace_id)
        cases = build_real_brain_cases(facts)
    else:
        cases = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    results: list[dict[str, object]] = []
    generator_fallbacks = 0
    for case in cases:
        request = QueryRequest(question=case["question"], top_k=args.top_k)
        _, contexts = await service.retrieve(request)
        try:
            generation = await service.generator.generate(
                request.question,
                contexts[:1],
                max_output_tokens=settings.max_output_tokens,
            )
            validate_grounding(generation.payload, contexts[:1])
            answer = generation.payload.answer
            citation_ids = generation.payload.citations
        except RuntimeError:
            if not contexts:
                raise
            generator_fallbacks += 1
            answer = contexts[0].text
            citation_ids = [contexts[0].fact_id]
        relevant = frozenset(case["relevant_fact_ids"])
        recall = recall_at_k(contexts, relevant, request.top_k)
        available = frozenset(context.fact_id for context in contexts)
        if judge:
            score = await judge.judge(
                request.question,
                answer,
                [context for context in contexts if context.fact_id in set(citation_ids)],
            )
        else:
            score = faithfulness(answer, citation_ids, available)
        results.append({"id": case["id"], "recall_at_k": recall, "faithfulness": score})
    summary = {
        "cases": len(results),
        "recall_at_k": sum(float(item["recall_at_k"]) for item in results) / len(results),
        "faithfulness": sum(float(item["faithfulness"]) for item in results) / len(results),
        "judge_model": settings.judge_model if judge else "local-citation-check",
        "generator_fallbacks": generator_fallbacks,
        "details": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
