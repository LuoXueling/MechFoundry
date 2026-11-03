"""Groq implementation of the provider-neutral generation interface."""

from __future__ import annotations

import base64
import os
from typing import Any

from .errors import provider_detail, APIConfigurationError, APIRequestError, APIResponseError
from .interfaces import GenerationRequest, GenerationResponse, TokenUsage


class GroqAPI:
    """Adapter for Groq's synchronous Chat Completions API."""

    default_model = "qwen/qwen3.8-27b"

    def __init__(self, api_key: str | None = None, *, client: Any | None = None) -> None:
        if client is not None:
            self._client = client
            return

        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        if not resolved_key:
            raise APIConfigurationError(
                "Groq is not configured; set GROQ_API_KEY or pass api_key"
            )
        try:
            from groq import Groq
        except ImportError as exc:
            raise APIConfigurationError(
                "groq is not installed; run: python -m pip install groq"
            ) from exc
        try:
            self._client = Groq(api_key=resolved_key)
        except Exception as exc:
            raise APIConfigurationError("failed to initialize the Groq client") from exc

    def generate_text(self, request: GenerationRequest) -> str:
        """Translate a normalized request to Groq and return response text."""

        return self.generate_response(request).text

    def generate_response(self, request: GenerationRequest) -> GenerationResponse:
        """Translate a normalized request and retain token accounting."""

        if request.top_k is not None:
            raise APIConfigurationError("Groq Chat Completions does not support top_k")

        messages: list[dict[str, Any]] = []
        if request.system_prompt is not None:
            messages.append({"role": "system", "content": request.system_prompt})

        if request.image is None:
            user_content: str | list[dict[str, Any]] = request.text or ""
        else:
            encoded = base64.b64encode(request.image.data).decode("ascii")
            user_content = []
            if request.text is not None:
                user_content.append({"type": "text", "text": request.text})
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{request.image.mime_type};base64,{encoded}"
                        )
                    },
                }
            )
        messages.append({"role": "user", "content": user_content})

        options: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": False,
        }
        optional_values = {
            "max_completion_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stop": list(request.stop_sequences)
            if request.stop_sequences is not None
            else None,
            "response_format": {"type": "json_object"} if request.response_json else None,
        }
        options.update(
            (name, value)
            for name, value in optional_values.items()
            if value is not None
        )

        try:
            response = self._client.chat.completions.create(**options)
        except Exception as exc:
            detail = provider_detail(exc)
            raise APIRequestError(
                f"Groq request failed ({type(exc).__name__})"
                + (f": {detail}" if detail else "")
            ) from exc
        try:
            text = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise APIResponseError("Groq returned an unexpected response") from exc
        if not isinstance(text, str) or not text.strip():
            raise APIResponseError("Groq returned no text")
        usage_data = getattr(response, "usage", None)
        usage = None
        if usage_data is not None:
            usage = TokenUsage(
                input_tokens=getattr(usage_data, "prompt_tokens", None),
                output_tokens=getattr(usage_data, "completion_tokens", None),
                total_tokens=getattr(usage_data, "total_tokens", None),
            )
        try:
            finish = response.choices[0].finish_reason
        except (AttributeError, IndexError, TypeError):
            finish = None
        return GenerationResponse(text=text, usage=usage,
                                  finish_reason=str(finish) if finish is not None else None)
