import asyncio
import os

from google.adk.agents import Agent
from google.adk.models import Gemini

from brain_rag.api import build_service
from brain_rag.models import QueryRequest

MODEL = os.getenv("ADK_MODEL", "gemini-3-flash-preview")


def retrieve_brain(question: str, top_k: int = 5) -> dict[str, object]:
    """Retrieve grounded Brain facts for a question and return fact IDs and text."""
    service = build_service()
    _, contexts = asyncio.run(
        service.retrieve(QueryRequest(question=question, top_k=max(1, min(top_k, 10))))
    )
    return {
        "facts": [
            {
                "fact_id": context.fact_id,
                "entity": context.entity_name,
                "section": context.section,
                "text": context.text,
            }
            for context in contexts
        ]
    }


root_agent = Agent(
    name="brain_rag_agent",
    model=Gemini(model=MODEL),
    instruction=(
        "Answer questions about TaskTree Brain. Always call retrieve_brain first, "
        "use only returned facts, and include the exact fact_id for every claim."
    ),
    tools=[retrieve_brain],
)
