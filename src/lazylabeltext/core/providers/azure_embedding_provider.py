"""Azure OpenAI embedding provider via the `openai` SDK's AzureOpenAI client.

Direct Azure REST — no langchain. langchain_openai's import chain pulls
in `transformers` (and through it `torch`) for token-counting utilities;
that's a heavy and irrelevant dep for an HTTP embedding call. Going
through openai.AzureOpenAI keeps the embedding path zero-torch.

Pairs naturally with AzureOpenAIProvider — same env-var auth, same
endpoint, same SSL / proxy story. Default deployment is `text-embedding-ada-002`.
"""

from __future__ import annotations

import logging

import numpy as np

from lazylabeltext.core.exceptions import EmbeddingProviderError

logger = logging.getLogger("lazylabeltext")


class AzureEmbeddingProvider:
    """Embedding provider via openai.AzureOpenAI (no langchain)."""

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

    _ENV_KEY_VARS = ("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY")
    _ENV_ENDPOINT_VARS = ("AZURE_OPENAI_ENDPOINT", "OPENAI_API_BASE")
    _ENV_VERSION_VARS = ("OPENAI_API_VERSION", "AZURE_OPENAI_API_VERSION")

    def _resolve_credentials(self) -> tuple[str, str, str]:
        """Same env-var fallback chain as AzureOpenAIProvider."""
        import os

        def _first_env(names: tuple[str, ...]) -> str:
            for name in names:
                v = os.environ.get(name)
                if v:
                    return v.strip()
            return ""

        if self.use_env_credentials:
            endpoint = self.azure_endpoint or _first_env(self._ENV_ENDPOINT_VARS)
            api_key = self.api_key or _first_env(self._ENV_KEY_VARS)
            api_version = self.api_version or _first_env(self._ENV_VERSION_VARS)
        else:
            endpoint = self.azure_endpoint or ""
            api_key = self.api_key or ""
            api_version = self.api_version

        if not api_version:
            raise EmbeddingProviderError(
                "azure-embeddings",
                "Azure API version missing — set 'API version' in Provider Settings "
                "or export OPENAI_API_VERSION.",
            )
        return endpoint, api_key, api_version

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise EmbeddingProviderError(
                "azure-embeddings",
                "openai package not installed (pip install openai httpx)",
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
            "api_version": api_version,
            "http_client": httpx_client,
            # max_retries=8 (vs SDK default 2) — same rationale as the LLM
            # client; gives transient 429 responses more chances to recover.
            "max_retries": 8,
        }
        if endpoint:
            kwargs["azure_endpoint"] = endpoint
        if api_key:
            kwargs["api_key"] = api_key

        try:
            self._client = AzureOpenAI(**kwargs)
        except Exception as e:
            extra = ""
            if self.use_env_credentials:
                import os

                names = list(self._ENV_KEY_VARS) + list(self._ENV_ENDPOINT_VARS) + list(
                    self._ENV_VERSION_VARS
                )
                visible = [n for n in names if os.environ.get(n)]
                extra = (
                    f"\n\nEnvironment seen by app: visible={visible or '(none)'}"
                )
            raise EmbeddingProviderError(
                "azure-embeddings", str(e) + extra
            ) from e
        return self._client

    def encode(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        try:
            response = client.embeddings.create(
                model=self.model_name,
                input=list(texts),
            )
        except Exception as e:
            raise EmbeddingProviderError("azure-embeddings", str(e)) from e
        return np.array([d.embedding for d in response.data], dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        client = self._get_client()
        try:
            response = client.embeddings.create(
                model=self.model_name,
                input=[text],
            )
        except Exception as e:
            raise EmbeddingProviderError("azure-embeddings", str(e)) from e
        return np.array(response.data[0].embedding, dtype=np.float32)

    def is_loaded(self) -> bool:
        return self._client is not None
