"""Centralized LLM configuration for the chatbot package.

This module exposes a LangChain-compatible ``llm`` instance and a low-level
OpenAI client that both support Azure OpenAI (preferred) and public OpenAI
credentials as a fallback. Azure deployments require the following
environment variables:

- ``AZURE_OPENAI_API_KEY``
- ``AZURE_OPENAI_ENDPOINT``
- ``AZURE_OPENAI_DEPLOYMENT`` (chat deployment name)
- ``AZURE_OPENAI_API_VERSION``

If ``AZURE_OPENAI_API_KEY`` is not set, the module falls back to the standard
OpenAI API using ``OPENAI_API_KEY`` and an optional ``OPENAI_CHAT_MODEL``
( defaults to ``gpt-4.1`` ).
"""

import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import AzureOpenAI, OpenAI

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
    """Create the LangChain LLM and raw client depending on the credentials."""

    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

    if os.getenv("AZURE_OPENAI_API_KEY"):
        api_key = _require_env("AZURE_OPENAI_API_KEY")
        endpoint = _require_env("AZURE_OPENAI_ENDPOINT")
        deployment = _require_env("AZURE_OPENAI_DEPLOYMENT")
        api_version = _require_env("AZURE_OPENAI_API_VERSION")
        embedding_deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", deployment
        )

        llm_instance = ChatOpenAI(
            openai_api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            openai_api_version=api_version,
            temperature=temperature,
        )

        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )

        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment=embedding_deployment,
            openai_api_version=api_version,
        )

        return {
            "llm": llm_instance,
            "client": client,
            "model_name": deployment,
            "embeddings": embeddings,
            "is_azure": True,
        }

    api_key = _require_env("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    llm_instance = ChatOpenAI(
        openai_api_key=api_key,
        model=model_name,
        temperature=temperature,
    )

    client = OpenAI(api_key=api_key)

    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        model=embedding_model,
    )

    return {
        "llm": llm_instance,
        "client": client,
        "model_name": model_name,
        "embeddings": embeddings,
        "is_azure": False,
    }


_config = _build_llm()
llm: ChatOpenAI = _config["llm"]
openai_client = _config["client"]
_model_name = _config["model_name"]
embedding_provider: OpenAIEmbeddings = _config["embeddings"]


def chat_completion(prompt: str, **kwargs: Any) -> str:
    """Helper to run a single-shot completion using the configured client."""

    response = openai_client.responses.create(
        model=_model_name,
        input=prompt,
        **kwargs,
    )
    return response.output_text
