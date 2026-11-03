from pathlib import Path

import pytest
from PIL import Image

from src.agents_wrapper import (
    APIResponseError,
    APIRequestError,
    AgentsWrapper,
    GenerationRequest,
    UnsupportedImageError,
)
import src.agents_wrapper.wrapper as wrapper_module


class RecordingAPI:
    def __init__(self, response="answer"):
        self.response = response
        self.requests = []

    def generate_text(self, request):
        self.requests.append(request)
        return self.response


def test_groq_is_default_provider(monkeypatch):
    class FakeGroq(RecordingAPI):
        default_model = "groq-default"

    monkeypatch.setattr(wrapper_module, "GroqAPI", FakeGroq)
    agent = AgentsWrapper()
    assert isinstance(agent.api, FakeGroq)
    assert agent.model == "groq-default"


def test_provider_default_model_is_used():
    api = RecordingAPI()
    api.default_model = "provider-default"
    assert AgentsWrapper(api).model == "provider-default"


def test_text_request_is_normalized():
    api = RecordingAPI()
    agent = AgentsWrapper(api, model="test-model")

    assert agent.generate("hello", temperature=0.2) == "answer"
    assert api.requests == [
        GenerationRequest(model="test-model", text="hello", temperature=0.2)
    ]


def test_path_image_and_text(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png data")
    api = RecordingAPI()

    AgentsWrapper(api).generate("inspect", image=image_path)

    request = api.requests[0]
    assert request.text == "inspect"
    assert request.image.data == b"png data"
    assert request.image.mime_type == "image/png"


def test_raw_bytes_require_mime_type():
    agent = AgentsWrapper(RecordingAPI())
    with pytest.raises(UnsupportedImageError, match="image_mime_type"):
        agent.generate(image=b"data")


def test_raw_bytes_image_only():
    api = RecordingAPI()
    AgentsWrapper(api).generate(image=b"data", image_mime_type="image/webp")
    assert api.requests[0].text is None
    assert api.requests[0].image.mime_type == "image/webp"


def test_pil_image_is_encoded_as_png():
    api = RecordingAPI()
    image = Image.new("RGB", (2, 3), "red")
    AgentsWrapper(api).generate(image=image)
    assert api.requests[0].image.data.startswith(b"\x89PNG")
    assert api.requests[0].image.mime_type == "image/png"


def test_requires_text_or_image():
    with pytest.raises(ValueError, match="at least one"):
        AgentsWrapper(RecordingAPI()).generate()


@pytest.mark.parametrize("value", ["", "   "])
def test_rejects_blank_text(value):
    with pytest.raises(ValueError, match="non-empty"):
        AgentsWrapper(RecordingAPI()).generate(value)


def test_generation_options_are_forwarded():
    api = RecordingAPI()
    AgentsWrapper(api).generate(
        "hello",
        system_prompt="be concise",
        max_output_tokens=20,
        top_p=0.9,
        top_k=10,
        stop_sequences=["END"],
    )
    request = api.requests[0]
    assert request.system_prompt == "be concise"
    assert request.max_output_tokens == 20
    assert request.top_p == 0.9
    assert request.top_k == 10
    assert request.stop_sequences == ("END",)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_output_tokens": 0}, "positive integer"),
        ({"temperature": -1}, "non-negative"),
        ({"top_p": 0}, r"\(0, 1]"),
        ({"top_k": True}, "positive integer"),
        ({"stop_sequences": "END"}, "sequence of strings"),
    ],
)
def test_invalid_generation_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AgentsWrapper(RecordingAPI()).generate("hello", **kwargs)


def test_invalid_custom_provider_response_is_rejected():
    with pytest.raises(APIResponseError, match="no text"):
        AgentsWrapper(RecordingAPI(response=None)).generate("hello")


def test_structured_generation_accepts_plain_json_and_fence():
    for response in ('{"value":1}', '```json\n{"value":1}\n```'):
        parsed, detail = AgentsWrapper(RecordingAPI(response=response)).generate_structured("task")
        assert parsed == {"value": 1}
        assert detail.text == response


@pytest.mark.parametrize("response", ["answer", "[]", "{} trailing"])
def test_structured_generation_rejects_non_object_json(response):
    with pytest.raises(APIResponseError):
        AgentsWrapper(RecordingAPI(response=response)).generate_structured("task")


def test_api_must_implement_protocol():
    with pytest.raises(TypeError, match="generate_text"):
        AgentsWrapper(object())


def test_transient_requests_are_retried(monkeypatch):
    class FlakyAPI(RecordingAPI):
        def generate_text(self, request):
            self.requests.append(request)
            if len(self.requests) < 3:
                try:
                    raise ConnectionError("offline")
                except ConnectionError as cause:
                    raise APIRequestError("request failed") from cause
            return "recovered"

    monkeypatch.setattr(wrapper_module.time, "sleep", lambda _: None)
    api = FlakyAPI()
    assert AgentsWrapper(api).generate("hello") == "recovered"
    assert len(api.requests) == 3


def test_permanent_request_error_is_not_retried():
    class BadRequestAPI(RecordingAPI):
        def generate_text(self, request):
            self.requests.append(request)
            try:
                error = RuntimeError("bad request")
                error.status_code = 400
                raise error
            except RuntimeError as cause:
                raise APIRequestError("request failed") from cause

    api = BadRequestAPI()
    with pytest.raises(APIRequestError):
        AgentsWrapper(api).generate("hello")
    assert len(api.requests) == 1

def test_provider_detail_surfaces_the_message_without_credentials():
    """Class name alone is unactionable; the raw exception may hold the request."""
    from src.agents_wrapper.errors import provider_detail

    class FakeGroqError(Exception):
        body = {"error": {"message": "Failed to validate JSON. Please adjust your prompt.",
                          "type": "invalid_request_error", "code": "json_validate_failed"}}

    assert provider_detail(FakeGroqError()) == (
        "Failed to validate JSON. Please adjust your prompt.")

    class LeakyError(Exception):
        body = {"error": {"message": "auth failed for key gsk_abcdef0123456789 on org x"}}

    detail = provider_detail(LeakyError())
    assert "gsk_abcdef0123456789" not in detail
    assert "<redacted>" in detail

    class MessageOnly(Exception):
        message = "  Rate limit reached.\n  Try again in 4m32s.  "

    assert provider_detail(MessageOnly()) == "Rate limit reached. Try again in 4m32s."
    assert provider_detail(Exception("plain")) == ""
    assert len(provider_detail(type("E", (Exception,), {"message": "x" * 500})())) == 240
