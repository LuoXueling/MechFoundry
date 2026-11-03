"""Public exceptions raised by :mod:`src.agents_wrapper`."""


class AgentsWrapperError(RuntimeError):
    """Base class for agent API failures."""


class APIConfigurationError(AgentsWrapperError):
    """The selected API client is not configured correctly."""


class APIRequestError(AgentsWrapperError):
    """The remote API request failed."""


class APIResponseError(AgentsWrapperError):
    """The API returned no usable text."""

    def __init__(self, message: str, *, raw_text: str | None = None, response=None):
        super().__init__(message)
        self.raw_text = raw_text
        self.response = response


class UnsupportedImageError(AgentsWrapperError, ValueError):
    """An image cannot be converted to a supported API representation."""


def provider_detail(exc: Exception, limit: int = 240) -> str:
    """The provider's own error text, without credentials or request contents.

    An exception class name alone ("BadRequestError") says nothing actionable,
    but the raw exception can carry the outgoing request.  Take only the
    provider's message field, and redact anything shaped like an API key.
    """
    import re

    message = None
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
        elif isinstance(error, str):
            message = error
    if not message:
        message = getattr(exc, "message", None)
    if not isinstance(message, str) or not message.strip():
        return ""
    redacted = re.sub(r"\b(?:gsk|sk|AIza)[A-Za-z0-9_\-]{8,}", "<redacted>", message)
    redacted = " ".join(redacted.split())
    return redacted[:limit]
