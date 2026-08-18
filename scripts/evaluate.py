import asyncio
import json
from pathlib import Path

from brain_rag.api import build_service
from brain_rag.config import get_settings
from brain_rag.eval import VertexFaithfulnessJudge, faithfulness, recall_at_k
from brain_rag.models import QueryRequest


async def main() -> None:
    settings = get_settings()
    service = build_service()
    await service.ingest()
    judge = VertexFaithfulnessJudge(settings) if settings.google_cloud_project else None
    cases = json.loads(Path("eval/golden.json").read_text(encoding="utf-8"))
    results: list[dict[str, object]] = []
    for case in cases:
        request = QueryRequest(question=case["question"], top_k=5)
        _, contexts = await service.retrieve(request)
        response = await service.query(request)
        relevant = frozenset(case["relevant_fact_ids"])
        recall = recall_at_k(contexts, relevant, request.top_k)
        available = frozenset(context.fact_id for context in contexts)
        if judge:
            citation_ids = {citation.fact_id for citation in response.citations}
            score = await judge.judge(
                request.question,
                response.answer,
                [context for context in contexts if context.fact_id in citation_ids],
            )
        else:
            score = faithfulness(
                response.answer,
                [citation.fact_id for citation in response.citations],
                available,
            )
        results.append({"id": case["id"], "recall_at_k": recall, "faithfulness": score})
    summary = {
        "cases": len(results),
        "recall_at_k": sum(float(item["recall_at_k"]) for item in results) / len(results),
        "faithfulness": sum(float(item["faithfulness"]) for item in results) / len(results),
        "judge_model": settings.judge_model if judge else "local-citation-check",
        "details": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
