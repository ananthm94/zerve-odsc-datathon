from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from analytics_agent.core.config import AgentConfig


def create_retriever(config: AgentConfig, k: int = 5):
    embeddings = OpenAIEmbeddings(model=config.embedding_model)
    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name=config.qdrant_collection_name,
        embedding=embeddings,
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        check_compatibility=False,
    )
    return vector_store.as_retriever(search_kwargs={"k": k})
