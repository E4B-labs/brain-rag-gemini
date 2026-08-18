import pytest
from httpx import ASGITransport, AsyncClient

from brain_rag import api


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    async with AsyncClient(transport=ASGITransport(app=api.app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_local_ingest_and_query_endpoints() -> None:
    api.build_service.cache_clear()
    async with AsyncClient(transport=ASGITransport(app=api.app), base_url="http://test") as client:
        ingest = await client.post("/v1/ingest")
        response = await client.post(
            "/v1/query",
            json={"question": "Gdzie Brain przechowuje obserwacje?", "top_k": 2},
        )
    assert ingest.status_code == 200
    assert ingest.json()["chunks_written"] >= 2
    assert response.status_code == 200
    assert response.json()["citations"]
