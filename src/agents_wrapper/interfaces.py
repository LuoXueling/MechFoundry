"""Provider-neutral request objects and API protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class ImageData:
    """Encoded image bytes and their Internet media type."""

    data: bytes
    mime_type: str


@dataclass(frozen=True)
class GenerationRequest:
    """A normalized request understood by API adapters."""

    model: str
    text: str | None = None
    image: ImageData | None = None
    system_prompt: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: tuple[str, ...] | None = None
    response_json: bool = False
    # How much chain of thought a model that thinks before answering may spend:
    # "none", "low", "high", or "max".  None leaves the provider's own default
    # alone, and an adapter that cannot express this ignores it -- so a caller
    # must still budget for reasoning unless it knows the provider honours it.
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class TokenUsage:
    """Provider-reported token accounting, when available."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class GenerationResponse:
    """Text plus optional usage metadata returned by an API adapter."""

    text: str
    usage: TokenUsage | None = None
    # Why the provider stopped ("length", "MAX_TOKENS", "stop", ...), when it says: a reply cut
    # off at the token budget is never mended into JSON (specialists._one_bracket_mend).
    finish_reason: str | None = None


@dataclass(frozen=True)
class ChatMessage:
    """Provider-neutral persisted chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    text: str | None = None
    image: ImageData | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    """A registered tool exposed to a model as JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """A model request to invoke a named registered tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Serializable result returned to a model after monitored execution."""

    tool_call_id: str
    name: str
    output: dict[str, Any]
    is_error: bool = False


@dataclass(frozen=True)
class AgentTurn:
    """One assistant turn with optional sequential tool requests."""

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage | None = None


@runtime_checkable
class TextGenerationAPI(Protocol):
    """Interface implemented by Gemini and future provider adapters."""

    def generate_text(self, request: GenerationRequest) -> str:
        """Generate and return plain text for one request."""
        ...
