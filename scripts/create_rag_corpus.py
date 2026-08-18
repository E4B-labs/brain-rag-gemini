"""Create a managed Vertex AI RAG Engine corpus without a Vector Search endpoint."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--display-name", default="brain-rag-gemini")
    args = parser.parse_args()

    import importlib

    vertexai = importlib.import_module("vertexai")
    rag = importlib.import_module("vertexai.preview.rag")
    vertexai.init(project=args.project, location=args.location)
    config_name = f"projects/{args.project}/locations/{args.location}/ragEngineConfig"
    rag.rag_data.update_rag_engine_config(
        rag_engine_config=rag.RagEngineConfig(
            name=config_name,
            rag_managed_db_config=rag.RagManagedDbConfig(mode=rag.Serverless()),
        )
    )
    corpus = rag.create_corpus(
        display_name=args.display_name,
        backend_config=rag.RagVectorDbConfig(
            vector_db=rag.RagManagedVertexVectorSearch()
        ),
    )
    print(corpus.name)


if __name__ == "__main__":
    main()
