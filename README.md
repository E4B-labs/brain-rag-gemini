# Brain RAG Gemini

Production-shaped, grounded RAG over the Brain memory used by TaskTree.
The RAG runtime is Google Cloud-native: Gemini embeddings and generation run on
Vertex AI, local development uses SQLite, the managed production path uses
Vertex AI RAG Engine, secrets run in Secret Manager, and the API deploys to
Cloud Run. Vector Search is an explicit opt-in adapter only.

TaskTree's source of truth is split by design: operational application data is
in Firestore, while Brain's entities, observations, and relations are in the
existing Supabase Postgres memory service. This repository reads Brain through
a read-only adapter and does not copy or mutate the source database.

## Architecture

```mermaid
flowchart LR
  T[TaskTree Brain\nSupabase Postgres] --> I[Ingest worker]
  I --> E[Gemini Embedding 001\nVertex AI]
  E --> V[Vertex AI RAG Engine\nmanaged corpus]
  V --> F[Grounded context\nmetadata in payload]
  C[Cloud Run FastAPI] --> Q[Query embedding]
  Q --> V
  C --> G[Gemini 3 Flash\nVertex AI]
  G -. hard questions .-> P[Gemini 3.1 Pro]
  C --> S[Secret Manager]
```

For local development, the same ports are backed by deterministic hash
embeddings and SQLite. This makes tests and `docker compose up` independent of
credentials while keeping the production adapters explicit.

## Current Google model IDs

The defaults are checked against the current Google Cloud documentation:

- `gemini-embedding-001`, configured to emit 768 dimensions for the index.
- `gemini-3-flash-preview` for normal answers and structured JSON.
- `gemini-3.1-pro-preview` as the current Gemini 3 Pro successor for difficult questions.
- `gemini-3.1-flash-lite` for the evaluation judge.

Gemini 3 Flash and Gemini 3.1 Pro support structured output and explicit context
caching. The model IDs are environment variables so a preview-to-stable model
transition does not require a code change.

## Local run

Requires Docker, or Python 3.12 with `uv`.

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Then:

```powershell
Invoke-RestMethod http://localhost:8080/health
Invoke-RestMethod -Method Post http://localhost:8080/v1/ingest
Invoke-RestMethod -Method Post -ContentType 'application/json' `
  -Body '{"question":"Gdzie Brain przechowuje obserwacje?","top_k":3}' `
  http://localhost:8080/v1/query
```

Without GCP or Supabase variables the API uses a tiny local fixture and local
SQLite, which is intended only for smoke testing. With `GOOGLE_CLOUD_PROJECT`
and Application Default Credentials, it uses Vertex AI. With
`TASKTREE_DATABASE_URL`, ingestion reads the TaskTree Brain workspace selected
by `BRAIN_WORKSPACE_ID`.

## Tests and quality

```powershell
uv sync --python 3.12
uv run pytest
uv run ruff check src tests scripts
uv run mypy src
```

Tests use fake embedding and generation ports. They cover chunking, alias
filtering, hybrid retrieval, entity/section filters, SQLite persistence,
grounding validation, routing, cost limits, ingestion batching, API behavior,
and evaluation metrics.

## Production setup

1. Set `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION=global`, and
   `GOOGLE_GENAI_USE_VERTEXAI=True`.
2. Set `TASKTREE_DATABASE_URL` outside the repository and grant the read-only
   role only the Brain tables needed by the adapter.
3. Create a managed RAG Engine corpus (no Vector Search endpoint) with
   `scripts/create_rag_corpus.py`, or use an existing corpus name.
4. Create the Cloud Run service account and Secret Manager secret with
   `scripts/deploy.ps1 -VectorBackend rag -RagCorpusName <corpus-resource>`.
   The command requires `TASKTREE_DATABASE_URL` in the process environment and
   streams the connection string directly into Secret Manager.
5. Run the one-time ingestion job and verify `/v1/query` with a real Vertex
   request. The service logs model, latency, token counts, and estimated cost
   for each embedding/generation call.

The local SQLite adapter is the default for tests and development. The `rag`
backend uploads chunk metadata with each document and retrieves from a managed
RAG Engine corpus, so no user-managed Vector Search endpoint is needed. Entity
and section filters are applied after retrieval before the grounding guardrail.
The `vertex` backend remains available for a separately approved, pre-provisioned
Vector Search index; neither the API nor the deployment script creates one.

Create a managed RAG Engine corpus:

```powershell
uv run python scripts/create_rag_corpus.py `
  --project brain-rag-gemini `
  --location us-central1 `
  --display-name brain-rag-gemini
```

Example deployment command after the corpus exists:

```powershell
$env:TASKTREE_DATABASE_URL = "<set outside the repository>"
./scripts/deploy.ps1 -ProjectId "brain-rag-gemini" `
  -VectorBackend rag `
  -RagCorpusName "projects/brain-rag-gemini/locations/us-central1/ragCorpora/<id>"
```

## Evaluation

`eval/golden.json` contains ten small gold questions. `scripts/evaluate.py` is a
starter runner for the local fixture; the production evaluation should run the
retriever and ask `gemini-3.1-flash-lite` to judge faithfulness. The metric
helpers are in `brain_rag.eval` and report recall@k and citation faithfulness.

## Optional Google ADK tool

The same retriever is exposed as a single ADK tool in `adk_agent/agent.py`:

```powershell
uv sync --extra adk
adk web adk_agent
```

The ADK wrapper is optional; the FastAPI API and its tests do not import ADK.

## Security

No service-account JSON, API key, Supabase password, or access token belongs in
this repository. Cloud Run receives secrets from Secret Manager. The answer
guardrail rejects any citation ID that was not returned by retrieval, and query
cost and output token limits are enforced before returning a result.
