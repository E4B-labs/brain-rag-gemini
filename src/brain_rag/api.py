import logging
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from .brain import StaticBrainSource, SupabaseBrainSource
from .config import get_settings
from .embeddings import GeminiEmbeddingProvider, LocalHashEmbedder
from .generation import LocalGroundedGenerator, VertexGeminiGenerator
from .models import IngestResponse, QueryRequest, QueryResponse
from .ports import BrainSource, Embedder, Generator, VectorStore
from .service import RagService
from .stores import RagEngineStore, SQLiteVectorStore, VertexVectorSearchStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("brain_rag.api")


def _demo_facts() -> list[dict[str, str]]:
    return [
        {
            "fact_id": "demo-1",
            "entity_id": "demo-tasktree",
            "entity_kind": "project",
            "entity_name": "TaskTree",
            "body": "TaskTree przechowuje dane operacyjne aplikacji w Firestore.",
            "section": "stack",
            "source": "import",
        },
        {
            "fact_id": "demo-2",
            "entity_id": "demo-brain",
            "entity_kind": "concept",
            "entity_name": "Brain",
            "body": "Brain przechowuje encje, obserwacje i relacje w bazie Supabase Postgres.",
            "section": "wdrozenie",
            "source": "import",
        },
    ]


@lru_cache(maxsize=1)
def build_service() -> RagService:
    from .models import BrainFact

    settings = get_settings()
    source: BrainSource
    if settings.brain_database_url:
        source = SupabaseBrainSource(settings.brain_database_url)
    else:
        source = StaticBrainSource([BrainFact.model_validate(fact) for fact in _demo_facts()])

    embedder: Embedder
    generator: Generator
    if settings.google_cloud_project:
        embedder = GeminiEmbeddingProvider(settings)
        generator = VertexGeminiGenerator(settings)
    else:
        embedder = LocalHashEmbedder()
        generator = LocalGroundedGenerator()

    store: VectorStore
    if settings.vector_store_backend == "vertex":
        project = settings.google_cloud_project
        index_name = settings.vertex_index_name
        endpoint_name = settings.vertex_index_endpoint_name
        deployed_index_id = settings.vertex_deployed_index_id
        if not all((project, index_name, endpoint_name, deployed_index_id)):
            raise ValueError(
                "Vertex vector store requires project, index, endpoint and deployed index ID"
            )
        assert project is not None
        assert index_name is not None
        assert endpoint_name is not None
        assert deployed_index_id is not None
        store = VertexVectorSearchStore(
            project=project,
            location=settings.vertex_vector_location,
            index_name=index_name,
            endpoint_name=endpoint_name,
            deployed_index_id=deployed_index_id,
            metadata_collection=settings.vertex_metadata_collection,
        )
    elif settings.vector_store_backend == "rag":
        project = settings.google_cloud_project
        if not project or not settings.rag_corpus_name:
            raise ValueError("RAG backend requires GOOGLE_CLOUD_PROJECT and RAG_CORPUS_NAME")
        store = RagEngineStore(
            project=project,
            location=settings.rag_location,
            corpus_name=settings.rag_corpus_name,
            staging_bucket=settings.rag_staging_bucket,
            min_similarity=settings.rag_min_similarity,
        )
    else:
        store = SQLiteVectorStore(settings.sqlite_path)
    return RagService(
        settings=settings,
        source=source,
        embedder=embedder,
        store=store,
        generator=generator,
    )


app = FastAPI(title="Brain RAG Gemini", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest() -> IngestResponse:
    try:
        return await build_service().ingest()
    except Exception as exc:
        logger.exception("ingest_failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        return await build_service().query(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("query_failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
