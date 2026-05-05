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

        try:
            httpx_client = httpx.Client(http2=self.http2, verify=self.verify_ssl)
        except Exception as e:
            # http2=True requires the h2 package; fall back to http/1.1 if missing.
            if self.http2:
                logger.warning(
                    "Falling back to HTTP/1.1 (h2 not available): %s", e
                )
                httpx_client = httpx.Client(http2=False, verify=self.verify_ssl)
            else:
                raise LLMProviderError("azure", f"httpx.Client failed: {e}") from e

        kwargs: dict = {
            "openai_api_version": self.api_version,
            "azure_deployment": self.model,
            "http_client": httpx_client,
        }
        if not self.use_env_credentials:
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.azure_endpoint:
                kwargs["azure_endpoint"] = self.azure_endpoint

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
