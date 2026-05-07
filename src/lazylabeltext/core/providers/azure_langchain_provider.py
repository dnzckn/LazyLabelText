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
import threading

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
        self._llm_lock = threading.Lock()

    # Env-var fallback chain matches what langchain-openai itself looks for,
    # so the env-mode toggle behaves the same as the bare AzureChatOpenAI()
    # constructor people use in standalone scripts.
    _ENV_KEY_VARS = ("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY")
    _ENV_ENDPOINT_VARS = ("AZURE_OPENAI_ENDPOINT", "OPENAI_API_BASE")
    _ENV_VERSION_VARS = ("OPENAI_API_VERSION", "AZURE_OPENAI_API_VERSION")

    def _resolve_credentials(self) -> tuple[str, str, str]:
        """Resolve (endpoint, api_key, api_version) — explicit fields first,
        then env vars (multiple common names), then empty.

        Empty values are NOT an error here: when use_env_credentials is on,
        we'd rather hand the empty values to AzureChatOpenAI and let its own
        validators raise a clear pydantic error than reject upfront. Only
        api_version is enforced because it's not consistently env-resolved.
        """
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
            raise LLMProviderError(
                "azure",
                "Azure API version missing — set 'API version' in Provider Settings "
                "or export OPENAI_API_VERSION.",
            )
        return endpoint, api_key, api_version

    def _visible_azure_env_summary(self) -> str:
        """List which Azure-relevant env vars are visible to this process.

        Used in error messages so the user can see at a glance whether the
        env var they exported is actually reaching the running app.
        """
        import os

        names = list(self._ENV_KEY_VARS) + list(self._ENV_ENDPOINT_VARS) + list(
            self._ENV_VERSION_VARS
        )
        visible = [n for n in names if os.environ.get(n)]
        missing = [n for n in names if n not in visible]
        return (
            f"visible to process: {visible or '(none)'}; "
            f"not set in process env: {missing}"
        )

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        with self._llm_lock:
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
            if self.use_env_credentials and (not endpoint or not api_key):
                logger.info(
                    "Azure env-credentials check — %s",
                    self._visible_azure_env_summary(),
                )

            try:
                httpx_client = httpx.Client(http2=self.http2, verify=self.verify_ssl)
            except Exception as e:
                if self.http2:
                    logger.warning(
                        "Falling back to HTTP/1.1 (h2 not available): %s", e
                    )
                    httpx_client = httpx.Client(http2=False, verify=self.verify_ssl)
                else:
                    raise LLMProviderError(
                        "azure", f"httpx.Client failed: {e}"
                    ) from e

            kwargs: dict = {
                "openai_api_version": api_version,
                "azure_deployment": self.model,
                "http_client": httpx_client,
            }
            if endpoint:
                kwargs["azure_endpoint"] = endpoint
            if api_key:
                kwargs["api_key"] = api_key

            try:
                self._llm = AzureChatOpenAI(**kwargs)
            except Exception as e:
                extra = ""
                if self.use_env_credentials:
                    extra = (
                        f"\n\nEnvironment seen by app: "
                        f"{self._visible_azure_env_summary()}"
                    )
                raise LLMProviderError("azure", str(e) + extra) from e
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
