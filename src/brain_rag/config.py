from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_cloud_project: str | None = Field(default=None, validation_alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str = Field(default="global", validation_alias="GOOGLE_CLOUD_LOCATION")
    brain_database_url: str | None = Field(default=None, validation_alias="TASKTREE_DATABASE_URL")
    brain_workspace_id: str = Field(default="demo-workspace", validation_alias="BRAIN_WORKSPACE_ID")

    vector_store_backend: str = Field(default="local", validation_alias="VECTOR_STORE_BACKEND")
    sqlite_path: str = Field(default="data/brain-rag.sqlite3", validation_alias="SQLITE_PATH")
    rag_corpus_name: str | None = Field(default=None, validation_alias="RAG_CORPUS_NAME")
    rag_location: str = Field(default="us-central1", validation_alias="RAG_LOCATION")
    rag_staging_bucket: str | None = Field(
        default=None, validation_alias="RAG_STAGING_BUCKET"
    )
    vertex_vector_location: str = Field(
        default="us-central1", validation_alias="VERTEX_VECTOR_LOCATION"
    )
    vertex_index_name: str | None = Field(default=None, validation_alias="VERTEX_INDEX_NAME")
    vertex_index_endpoint_name: str | None = Field(
        default=None, validation_alias="VERTEX_INDEX_ENDPOINT_NAME"
    )
    vertex_deployed_index_id: str | None = Field(
        default=None, validation_alias="VERTEX_DEPLOYED_INDEX_ID"
    )
    vertex_metadata_collection: str = Field(
        default="brain_rag_chunks", validation_alias="VERTEX_METADATA_COLLECTION"
    )

    embedding_model: str = Field(default="gemini-embedding-001", validation_alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=768, validation_alias="EMBEDDING_DIMENSION")
    generation_model: str = Field(
        default="gemini-3-flash-preview", validation_alias="GENERATION_MODEL"
    )
    fallback_model: str = Field(default="gemini-3.1-pro-preview", validation_alias="FALLBACK_MODEL")
    judge_model: str = Field(default="gemini-3.1-flash-lite", validation_alias="JUDGE_MODEL")
    enable_context_cache: bool = Field(default=True, validation_alias="ENABLE_CONTEXT_CACHE")
    max_output_tokens: int = Field(default=800, validation_alias="MAX_OUTPUT_TOKENS")
    max_query_cost_usd: float = Field(default=0.05, validation_alias="MAX_QUERY_COST_USD")
    rag_min_similarity: float = Field(default=0.55, validation_alias="RAG_MIN_SIMILARITY")
    rag_candidate_multiplier: int = Field(
        default=8, validation_alias="RAG_CANDIDATE_MULTIPLIER"
    )
    max_query_chars: int = 20_000

    def validate_runtime(self) -> None:
        if self.vector_store_backend not in {"local", "rag", "vertex"}:
            raise ValueError("VECTOR_STORE_BACKEND must be 'local', 'rag', or 'vertex'")
        if self.embedding_dimension <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("MAX_OUTPUT_TOKENS must be positive")
        if self.max_query_cost_usd <= 0:
            raise ValueError("MAX_QUERY_COST_USD must be positive")
        if not 0 <= self.rag_min_similarity < 1:
            raise ValueError("RAG_MIN_SIMILARITY must be between 0 and 1")
        if self.rag_candidate_multiplier <= 0:
            raise ValueError("RAG_CANDIDATE_MULTIPLIER must be positive")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_runtime()
    return settings
