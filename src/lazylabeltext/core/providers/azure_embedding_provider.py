"""Azure OpenAI embedding provider via langchain_openai.AzureOpenAIEmbeddings.

Pairs naturally with AzureLangChainProvider — same env-var auth, same
endpoint, same SSL / proxy story. Default deployment is `text-embedding-ada-002`.
"""

from __future__ import annotations

import logging

import numpy as np

from lazylabeltext.core.exceptions import EmbeddingProviderError

logger = logging.getLogger("lazylabeltext")


class AzureEmbeddingProvider:
    """Embedding provider using Azure OpenAI via LangChain."""

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: str | None = None,
        api_version: str = "2024-08-01-preview",
        azure_endpoint: str | None = None,
        verify_ssl: bool = True,
        http2: bool = True,
        use_env_credentials: bool = True,
    ) -> None:
        self.model_name = model_name  # azure deployment name
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.verify_ssl = verify_ssl
        self.http2 = http2
        self.use_env_credentials = use_env_credentials
        self._client = None

    def _resolve_credentials(self) -> tuple[str, str, str]:
        """Mirror AzureLangChainProvider's resolution: explicit fields first,
        env vars as fallback. Avoids depending on langchain-openai's version-
        specific env-var handling.
        """
        import os

        if self.use_env_credentials:
            endpoint = self.azure_endpoint or os.environ.get(
                "AZURE_OPENAI_ENDPOINT", ""
            )
            api_key = self.api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
            api_version = self.api_version or os.environ.get(
                "OPENAI_API_VERSION", ""
            )
        else:
            endpoint = self.azure_endpoint or ""
            api_key = self.api_key or ""
            api_version = self.api_version

        missing = []
        if not endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not api_version:
            missing.append("API version")
        if missing:
            raise EmbeddingProviderError(
                "azure-embeddings",
                "Azure credentials missing: " + "; ".join(missing),
            )
        return endpoint, api_key, api_version

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from langchain_openai import AzureOpenAIEmbeddings
        except ImportError as e:
            raise EmbeddingProviderError(
                "azure-embeddings",
                "langchain-openai not installed (pip install langchain-openai httpx)",
            ) from e
        try:
            import httpx
        except ImportError as e:
            raise EmbeddingProviderError(
                "azure-embeddings", "httpx not installed"
            ) from e

        endpoint, api_key, api_version = self._resolve_credentials()

        try:
            httpx_client = httpx.Client(http2=self.http2, verify=self.verify_ssl)
        except Exception as e:
            if self.http2:
                logger.warning(
                    "Falling back to HTTP/1.1 (h2 not available): %s", e
                )
                httpx_client = httpx.Client(http2=False, verify=self.verify_ssl)
            else:
                raise EmbeddingProviderError(
                    "azure-embeddings", f"httpx.Client failed: {e}"
                ) from e

        kwargs: dict = {
            "openai_api_version": api_version,
            "azure_deployment": self.model_name,
            "azure_endpoint": endpoint,
            "api_key": api_key,
            "http_client": httpx_client,
        }

        try:
            self._client = AzureOpenAIEmbeddings(**kwargs)
        except Exception as e:
            raise EmbeddingProviderError("azure-embeddings", str(e)) from e
        return self._client

    def encode(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        try:
            vectors = client.embed_documents(list(texts))
        except Exception as e:
            raise EmbeddingProviderError("azure-embeddings", str(e)) from e
        return np.array(vectors, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        client = self._get_client()
        try:
            vector = client.embed_query(text)
        except Exception as e:
            raise EmbeddingProviderError("azure-embeddings", str(e)) from e
        return np.array(vector, dtype=np.float32)

    def is_loaded(self) -> bool:
        return self._client is not None
