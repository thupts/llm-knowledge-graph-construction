"""Centralized LLM configuration for the chatbot package.

This module exposes a LangChain-compatible ``llm`` instance, an embeddings
provider, and a low-level Azure OpenAI client.

Required environment variables:

- ``AZURE_OPENAI_API_KEY``
- ``AZURE_OPENAI_ENDPOINT``
- ``AZURE_OPENAI_DEPLOYMENT`` (chat deployment name)
- ``AZURE_OPENAI_API_VERSION``
- ``AZURE_OPENAI_EMBEDDING_DEPLOYMENT`` (embedding deployment name)
"""

import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import AzureOpenAI

load_dotenv()

__all__ = ["llm", "openai_client", "embedding_provider", "chat_completion"]


def _require_env(var_name: str) -> str:
    """Return the value for ``var_name`` or raise with a helpful message."""
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {var_name}. "
            "Check your .env configuration before starting the application."
        )
    return value


def _build_llm() -> Dict[str, Any]:
    """Create the LangChain LLM, embeddings, and Azure OpenAI client."""
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

    # Azure settings
    api_key = _require_env("AZURE_OPENAI_API_KEY")
    endpoint = _require_env("AZURE_OPENAI_ENDPOINT")
    deployment = _require_env("AZURE_OPENAI_DEPLOYMENT")
    chat_api_version = _require_env("AZURE_OPENAI_CHAT_API_VERSION")
    embedding_deployment = _require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    embedding_api_version = _require_env("AZURE_OPENAI_EMBEDDING_API_VERSION")

    # LangChain LLM (chat model)
    llm_instance = ChatOpenAI(
        openai_api_key=api_key,
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        openai_api_version=chat_api_version,
        temperature=temperature,
    )

    # Raw Azure client (chat)
    client = AzureOpenAI(
        api_key=api_key,
        api_version=chat_api_version,
        azure_endpoint=endpoint,
    )

    # Embeddings provider 
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        azure_endpoint=endpoint,
        azure_deployment=embedding_deployment,
        openai_api_version=embedding_api_version,
    )

    return {
        "llm": llm_instance,
        "client": client,
        "model_name": deployment,
        "embeddings": embeddings,
    }



_config = _build_llm()
llm: ChatOpenAI = _config["llm"]
openai_client: AzureOpenAI = _config["client"]
_model_name: str = _config["model_name"]
embedding_provider: OpenAIEmbeddings = _config["embeddings"]


def chat_completion(prompt: str, **kwargs: Any) -> str:
    """Helper to run a single-shot completion using the configured Azure client."""
    response = openai_client.responses.create(
        model=_model_name,
        input=prompt,
        **kwargs,
    )
    return response.output_text

if __name__ == "__main__":
    print("✅ llm.py loaded successfully")
    print("Chat model deployment:", _model_name)

    # Test chat completion
    reply = chat_completion("Hello, who are you?")
    print("ChatCompletion test:", reply[:200], "...")

    # Test embeddings
    test_text = "Neo4j is a graph database"
    vector = embedding_provider.embed_query(test_text)
    print("Embedding length:", len(vector))
    print("First 5 dims:", vector[:5])
