"""Gemini implementation of the provider-neutral generation interface."""

from __future__ import annotations

import os
from typing import Any

from .errors import provider_detail, APIConfigurationError, APIRequestError, APIResponseError
from .interfaces import GenerationRequest, GenerationResponse, TokenUsage


class GeminiAPI:
    """Adapter for the ``google-genai`` synchronous Chat API."""

    default_model = "gemini-2.5-flash"

    def __init__(self, api_key: str | None = None, *, client: Any | None = None) -> None:
        if client is not None:
            self._client = client
            self._types = None
            return

        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise APIConfigurationError(
                "Gemini is not configured; set GEMINI_API_KEY or pass api_key"
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise APIConfigurationError(
                "google-genai is not installed; run: python -m pip install google-genai"
            ) from exc

        self._types = types
        try:
            self._client = genai.Client(api_key=resolved_key)
        except Exception as exc:
            raise APIConfigurationError("failed to initialize the Gemini client") from exc

    def generate_text(self, request: GenerationRequest) -> str:
        """Translate a normalized request to Gemini and return response text."""

        return self.generate_response(request).text

    def generate_response(self, request: GenerationRequest) -> GenerationResponse:
        """Translate a normalized request and retain token accounting."""

        types = self._get_types()
        contents: str | list[Any]
        if request.image is None:
            contents = request.text or ""
        else:
            contents = [
                types.Part.from_bytes(
                    data=request.image.data,
                    mime_type=request.image.mime_type,
                )
            ]
            if request.text is not None:
                contents.append(request.text)

        config_values = {
            # The workflow executes tools itself. Disable SDK-side AFC and use
            # Chat.send_message so adding declarations later remains warning-free.
            "automatic_function_calling": {"disable": True},
            "system_instruction": request.system_prompt,
            "max_output_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "stop_sequences": list(request.stop_sequences)
            if request.stop_sequences is not None
            else None,
            "response_mime_type": "application/json" if request.response_json else None,
        }
        config_values = {
            name: value for name, value in config_values.items() if value is not None
        }
        config = types.GenerateContentConfig(**config_values)

        try:
            chat = self._client.chats.create(
                model=request.model,
                config=config,
            )
            response = chat.send_message(contents)
            text = response.text
        except Exception as exc:
            # Keep credentials and provider request details out of the public
            # message while retaining the original exception as the cause.
            detail = provider_detail(exc)
            raise APIRequestError(
                f"Gemini request failed ({type(exc).__name__})"
                + (f": {detail}" if detail else "")
            ) from exc
        if not isinstance(text, str) or not text.strip():
            raise APIResponseError(
                "Gemini returned no text; the response may have been blocked or empty"
            )
        usage_data = getattr(response, "usage_metadata", None)
        usage = None
        if usage_data is not None:
            # Gemini reports a thinking model's "thoughts" beside the candidates
            # rather than inside them, while total_token_count counts both, and
            # the OpenAI-shaped providers fold reasoning into completion tokens.
            # Counting thoughts as output keeps the three comparable and keeps
            # input + output reconciling with the total.
            answered = getattr(usage_data, "candidates_token_count", None)
            thoughts = getattr(usage_data, "thoughts_token_count", None)
            output_tokens = (
                None
                if answered is None and thoughts is None
                else (answered or 0) + (thoughts or 0)
            )
            usage = TokenUsage(
                input_tokens=getattr(usage_data, "prompt_token_count", None),
                output_tokens=output_tokens,
                total_tokens=getattr(usage_data, "total_token_count", None),
            )
        try:
            finish = response.candidates[0].finish_reason
            finish = getattr(finish, "name", finish)
        except (AttributeError, IndexError, TypeError):
            finish = None
        return GenerationResponse(text=text, usage=usage,
                                  finish_reason=str(finish) if finish is not None else None)

    def _get_types(self):
        if self._types is not None:
            return self._types
        try:
            from google.genai import types
        except ImportError as exc:
            raise APIConfigurationError("google-genai is not installed") from exc
        return types
