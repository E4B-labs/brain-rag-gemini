param(
  [Parameter(Mandatory = $true)][string]$ProjectId,
  [string]$Region = "europe-west1",
  [string]$Service = "brain-rag-gemini",
  [string]$Repository = "brain-rag",
  [Parameter(Mandatory = $true)][string]$VertexIndexName,
  [Parameter(Mandatory = $true)][string]$VertexIndexEndpointName,
  [Parameter(Mandatory = $true)][string]$VertexDeployedIndexId,
  [string]$VertexLocation = "us-central1"
)

$ErrorActionPreference = "Stop"
$runtimeSa = "brain-rag-runtime@$ProjectId.iam.gserviceaccount.com"
$image = "$Region-docker.pkg.dev/$ProjectId/$Repository/$Service:manual"

gcloud config set project $ProjectId
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com aiplatform.googleapis.com firestore.googleapis.com secretmanager.googleapis.com

gcloud artifacts repositories describe $Repository --location=$Region 2>$null
if ($LASTEXITCODE -ne 0) {
  gcloud artifacts repositories create $Repository --repository-format=docker --location=$Region
}

gcloud iam service-accounts describe $runtimeSa 2>$null
if ($LASTEXITCODE -ne 0) {
  gcloud iam service-accounts create brain-rag-runtime --display-name="Brain RAG Cloud Run runtime"
}

gcloud projects add-iam-policy-binding $ProjectId --member="serviceAccount:$runtimeSa" --role="roles/aiplatform.user"
gcloud projects add-iam-policy-binding $ProjectId --member="serviceAccount:$runtimeSa" --role="roles/datastore.user"
if (-not $env:TASKTREE_DATABASE_URL) {
  throw "TASKTREE_DATABASE_URL is missing from the local environment. It is used only to create the Secret Manager version and is never committed."
}
gcloud secrets describe tasktree-database-url 2>$null
if ($LASTEXITCODE -ne 0) {
  gcloud secrets create tasktree-database-url --replication-policy=automatic
}
$env:TASKTREE_DATABASE_URL | gcloud secrets versions add tasktree-database-url --data-file=-
gcloud secrets add-iam-policy-binding tasktree-database-url --member="serviceAccount:$runtimeSa" --role="roles/secretmanager.secretAccessor"

gcloud builds submit --tag $image
gcloud run deploy $Service --image $image --region $Region --service-account $runtimeSa --no-allow-unauthenticated --set-env-vars="GOOGLE_CLOUD_PROJECT=$ProjectId,GOOGLE_CLOUD_LOCATION=global,GOOGLE_GENAI_USE_VERTEXAI=True,VECTOR_STORE_BACKEND=vertex,VERTEX_INDEX_NAME=$VertexIndexName,VERTEX_INDEX_ENDPOINT_NAME=$VertexIndexEndpointName,VERTEX_DEPLOYED_INDEX_ID=$VertexDeployedIndexId,VERTEX_VECTOR_LOCATION=$VertexLocation" --set-secrets="TASKTREE_DATABASE_URL=tasktree-database-url:latest"
