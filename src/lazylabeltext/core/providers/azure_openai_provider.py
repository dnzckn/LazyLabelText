"""Azure OpenAI LLM provider via the `openai` SDK's AzureOpenAI client.

Goes direct to the Azure REST endpoint — no langchain. Avoiding langchain
matters for two reasons: it pulls in `transformers` (and through it,
`torch`) at import time for token-counting utilities we don't need; and
its message wrapper layer is dead weight when our message shape is just
[{role, content}, ...].

Same env-var fallback chain as the embedding sibling: explicit fields
first, then env vars (multiple common names), then empty.
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


class AzureOpenAIProvider:
    """LLM classification via openai.AzureOpenAI (no langchain)."""

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
        self.model = model  # azure deployment name
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.verify_ssl = verify_ssl
        self.http2 = http2
        self.use_env_credentials = use_env_credentials
        self._client = None
        self._client_lock = threading.Lock()

    _ENV_KEY_VARS = ("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY")
    _ENV_ENDPOINT_VARS = ("AZURE_OPENAI_ENDPOINT", "OPENAI_API_BASE")
    _ENV_VERSION_VARS = ("OPENAI_API_VERSION", "AZURE_OPENAI_API_VERSION")

    def _resolve_credentials(self) -> tuple[str, str, str]:
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

    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client

            try:
                from openai import AzureOpenAI
            except ImportError as e:
                raise LLMProviderError(
                    "azure", "openai package not installed (pip install openai httpx)"
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
                "api_version": api_version,
                "http_client": httpx_client,
                # max_retries=8 (vs SDK default 2) — gives transient 429
                # rate-limit responses more chances to recover.
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
                    extra = (
                        f"\n\nEnvironment seen by app: "
                        f"{self._visible_azure_env_summary()}"
                    )
                raise LLMProviderError("azure", str(e) + extra) from e
        return self._client

    def complete(self, prompt: str, max_tokens: int = 4096) -> str:
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            raise LLMProviderError("azure", str(e)) from e
        return response.choices[0].message.content or ""

    def classify(
        self, chunk_text: str, categories: list[Category]
    ) -> ClassificationResult:
        client = self._get_client()
        system_prompt = build_classification_system_prompt(categories)
        user_message = f"Classify this text chunk:\n\n{chunk_text}"

        try:
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                logprobs=True,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
        except Exception as e:
            raise LLMProviderError("azure", str(e)) from e

        result = parse_classification_response(
            response.choices[0].message.content or "", categories
        )
        try:
            content = response.choices[0].logprobs.content  # type: ignore[union-attr]
            if content:
                lp_values = [tok.logprob for tok in content if tok.logprob is not None]
                if lp_values:
                    result.avg_logprob = sum(lp_values) / len(lp_values)
        except Exception:
            pass
        return result

    def test_connection(self) -> tuple[bool, str]:
        try:
            client = self._get_client()
            client.chat.completions.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            return (
                True,
                f"Connected to deployment '{self.model}' (api {self.api_version})",
            )
        except Exception as e:
            return False, str(e)
