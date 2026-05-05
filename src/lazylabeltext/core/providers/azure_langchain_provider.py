"""Azure OpenAI LLM provider via langchain_openai.AzureChatOpenAI.

Designed for corporate / Azure deployments where:
  - Auth lives in env vars (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)
    so secrets never enter app config.
  - A custom httpx.Client is required for http2 / non-default SSL verify
    (self-signed CA, MITM proxy, etc.).
  - The user picks deployment name + api_version per-session in the UI,
    so swapping models is a settings change, not a code change.
"""

from __future__ import annotations

import logging

from lazylabeltext.core.exceptions import LLMProviderError
from lazylabeltext.core.models import Category, ClassificationResult
from lazylabeltext.core.providers._classification import (
    build_classification_system_prompt,
    parse_classification_response,
)

logger = logging.getLogger("lazylabeltext")


class AzureLangChainProvider:
    """LLM classification using Azure OpenAI via LangChain's AzureChatOpenAI."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        api_version: str = "2024-08-01-preview",
        azure_endpoint: str | None = None,
        verify_ssl: bool = True,
        http2: bool = True,
        use_env_credentials: bool = True,
    ) -> None:
        self.api_key = api_key
        self.model = model  # azure deployment name (often == openai model name)
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.verify_ssl = verify_ssl
        self.http2 = http2
        self.use_env_credentials = use_env_credentials
        self._llm = None

    def _resolve_credentials(self) -> tuple[str, str, str]:
        """Resolve (endpoint, api_key, api_version) using explicit fields first
        and env vars as fallback. Raises if a required value is missing.

        Doing this in our code rather than relying on langchain_openai's
        validate_environment hook means env-mode works the same across all
        langchain versions, and missing env vars produce a clear error.
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
            missing.append("AZURE_OPENAI_ENDPOINT (or set Azure endpoint in settings)")
        if not api_key:
            missing.append("AZURE_OPENAI_API_KEY (or paste the key in settings)")
        if not api_version:
            missing.append("API version (set in settings)")
        if missing:
            raise LLMProviderError(
                "azure",
                "Azure credentials missing: " + "; ".join(missing),
            )
        return endpoint, api_key, api_version

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as e:
            raise LLMProviderError(
                "azure",
                "langchain-openai not installed (pip install langchain-openai httpx)",
            ) from e
        try:
            import httpx
        except ImportError as e:
            raise LLMProviderError(
                "azure", "httpx not installed (pip install httpx)"
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
                raise LLMProviderError("azure", f"httpx.Client failed: {e}") from e

        kwargs: dict = {
            "openai_api_version": api_version,
            "azure_deployment": self.model,
            "azure_endpoint": endpoint,
            "api_key": api_key,
            "http_client": httpx_client,
        }

        try:
            self._llm = AzureChatOpenAI(**kwargs)
        except Exception as e:
            raise LLMProviderError("azure", str(e)) from e
        return self._llm

    def complete(self, prompt: str, max_tokens: int = 4096) -> str:
        llm = self._get_llm()
        try:
            result = llm.invoke(prompt)
        except Exception as e:
            raise LLMProviderError("azure", str(e)) from e
        return getattr(result, "content", str(result))

    def classify(
        self, chunk_text: str, categories: list[Category]
    ) -> ClassificationResult:
        llm = self._get_llm()
        system_prompt = build_classification_system_prompt(categories)
        user_message = f"Classify this text chunk:\n\n{chunk_text}"

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            result = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message),
                ]
            )
        except ImportError:
            # Fall back to a single concatenated prompt if langchain_core isn't
            # importable for some reason (unlikely — it's a transitive dep).
            try:
                result = llm.invoke(f"{system_prompt}\n\n{user_message}")
            except Exception as e:
                raise LLMProviderError("azure", str(e)) from e
        except Exception as e:
            raise LLMProviderError("azure", str(e)) from e

        text = getattr(result, "content", str(result))
        return parse_classification_response(text, categories)

    def test_connection(self) -> tuple[bool, str]:
        try:
            llm = self._get_llm()
            llm.invoke("Say OK")
            return (
                True,
                f"Connected to deployment '{self.model}' (api {self.api_version})",
            )
        except Exception as e:
            return False, str(e)
