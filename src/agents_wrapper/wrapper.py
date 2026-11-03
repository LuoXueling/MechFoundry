"""Provider-neutral text and image agent wrapper."""

from __future__ import annotations

import io
import json
import mimetypes
import time
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from .errors import APIResponseError, UnsupportedImageError
from .groq import GroqAPI
from .interfaces import GenerationRequest, GenerationResponse, ImageData, TextGenerationAPI

ImageInput = str | Path | bytes | Image.Image

_SUPPORTED_IMAGE_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}


class AgentsWrapper:
    """Submit stateless text or image requests through an API adapter."""

    def __init__(
        self,
        api: TextGenerationAPI | None = None,
        model: str | None = None,
        max_request_attempts: int = 3,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        if api is None:
            api = GroqAPI()
        if model is None:
            model = getattr(api, "default_model", GroqAPI.default_model)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(api, TextGenerationAPI):
            raise TypeError("api must implement generate_text(request)")
        self.api = api
        self.model = model.strip()
        if max_request_attempts < 1:
            raise ValueError("max_request_attempts must be positive")
        self.max_request_attempts = max_request_attempts
        self.retry_backoff_seconds = retry_backoff_seconds

    def generate(
        self,
        text: str | None = None,
        *,
        image: ImageInput | None = None,
        image_mime_type: str | None = None,
        system_prompt: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: Sequence[str] | None = None,
    ) -> str:
        """Generate text from text, an image, or both."""

        normalized_text = self._optional_nonempty_text(text, "text")
        normalized_system = self._optional_nonempty_text(
            system_prompt, "system_prompt"
        )
        if normalized_text is None and image is None:
            raise ValueError("at least one of text or image must be supplied")

        self._validate_generation_options(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        normalized_stops = self._normalize_stop_sequences(stop_sequences)
        normalized_image = (
            self._normalize_image(image, image_mime_type) if image is not None else None
        )
        request = GenerationRequest(
            model=self.model,
            text=normalized_text,
            image=normalized_image,
            system_prompt=normalized_system,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=normalized_stops,
        )
        result = self._request(request)
        if not isinstance(result.text, str) or not result.text.strip():
            raise APIResponseError("the API returned no text")
        return result.text

    def generate_structured(
        self,
        text: str,
        *,
        image: ImageInput | None = None,
        image_mime_type: str | None = None,
        system_prompt: str | None = None,
        max_output_tokens: int = 256,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> tuple[dict, GenerationResponse]:
        """Return a compact JSON object and provider token accounting.

        Markdown fences are tolerated, but prose around the object is rejected so
        callers never need another API call merely to parse a response.
        """

        normalized_text = self._optional_nonempty_text(text, "text")
        normalized_image = (
            self._normalize_image(image, image_mime_type) if image is not None else None
        )
        request = GenerationRequest(
            model=self.model,
            text=normalized_text,
            image=normalized_image,
            system_prompt=self._optional_nonempty_text(system_prompt, "system_prompt"),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            response_json=True,
            reasoning_effort=reasoning_effort,
        )
        response = self._request(request)
        raw = response.text.strip()
        if raw.startswith("```") and raw.endswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1]).strip()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise APIResponseError(
                "the API did not return a single JSON value",
                raw_text=response.text, response=response,
            ) from exc
        if not isinstance(value, dict):
            raise APIResponseError(
                "the API JSON response must be an object",
                raw_text=response.text, response=response,
            )
        return value, response

    def _request(self, request: GenerationRequest) -> GenerationResponse:
        from .errors import APIRequestError

        for attempt in range(self.max_request_attempts):
            try:
                detailed = getattr(self.api, "generate_response", None)
                if callable(detailed):
                    response = detailed(request)
                    if not isinstance(response, GenerationResponse):
                        raise APIResponseError("generate_response returned an invalid value")
                    return response
                return GenerationResponse(text=self.api.generate_text(request))
            except APIRequestError as exc:
                if attempt + 1 >= self.max_request_attempts or not self._is_transient(exc):
                    raise
                time.sleep(self.retry_backoff_seconds * (2 ** attempt))
        raise AssertionError("unreachable")

    @staticmethod
    def _is_transient(error: Exception) -> bool:
        cause = error.__cause__ or error
        if isinstance(cause, (ConnectionError, TimeoutError)):
            return True
        status = getattr(cause, "status_code", None)
        if status is None:
            response = getattr(cause, "response", None)
            status = getattr(response, "status_code", None)
        if status in (408, 409, 425, 429) or isinstance(status, int) and status >= 500:
            return True
        name = type(cause).__name__.lower()
        return any(token in name for token in ("timeout", "ratelimit", "connection", "internalserver"))

    @staticmethod
    def _optional_nonempty_text(value: str | None, name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string when supplied")
        return value

    @staticmethod
    def _validate_generation_options(
        *,
        max_output_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
    ) -> None:
        if max_output_tokens is not None and (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or temperature < 0
        ):
            raise ValueError("temperature must be a non-negative number")
        if top_p is not None and (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not 0 < top_p <= 1
        ):
            raise ValueError("top_p must be in (0, 1]")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer")

    @staticmethod
    def _normalize_stop_sequences(
        values: Sequence[str] | None,
    ) -> tuple[str, ...] | None:
        if values is None:
            return None
        if isinstance(values, (str, bytes)):
            raise ValueError("stop_sequences must be a sequence of strings")
        result = tuple(values)
        if not result or any(not isinstance(item, str) or not item for item in result):
            raise ValueError("stop_sequences must contain non-empty strings")
        return result

    @classmethod
    def _normalize_image(
        cls, image: ImageInput, explicit_mime_type: str | None
    ) -> ImageData:
        if explicit_mime_type is not None:
            explicit_mime_type = explicit_mime_type.lower().strip()
        if isinstance(image, bytes):
            if not explicit_mime_type:
                raise UnsupportedImageError(
                    "image_mime_type is required when image is raw bytes"
                )
            data = image
            mime_type = explicit_mime_type
        elif isinstance(image, (str, Path)):
            path = Path(image)
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise UnsupportedImageError(f"cannot read image {path}: {exc}") from exc
            mime_type = explicit_mime_type or mimetypes.guess_type(path.name)[0]
            if mime_type is None:
                raise UnsupportedImageError(
                    f"cannot infer the MIME type for image {path}; supply image_mime_type"
                )
        elif isinstance(image, Image.Image):
            mime_type = explicit_mime_type or Image.MIME.get(image.format or "", "image/png")
            output = io.BytesIO()
            format_name = {
                "image/gif": "GIF",
                "image/jpeg": "JPEG",
                "image/png": "PNG",
                "image/webp": "WEBP",
            }.get(mime_type)
            if format_name is None:
                raise UnsupportedImageError(f"unsupported image MIME type: {mime_type}")
            try:
                image.save(output, format=format_name)
            except (OSError, ValueError) as exc:
                raise UnsupportedImageError(f"cannot encode image: {exc}") from exc
            data = output.getvalue()
        else:
            raise UnsupportedImageError(
                "image must be a path, bytes, or PIL.Image.Image"
            )

        normalized_mime = mime_type.lower().strip()
        if normalized_mime not in _SUPPORTED_IMAGE_TYPES:
            raise UnsupportedImageError(
                f"unsupported image MIME type: {normalized_mime}"
            )
        if not data:
            raise UnsupportedImageError("image data is empty")
        return ImageData(data=data, mime_type=normalized_mime)
