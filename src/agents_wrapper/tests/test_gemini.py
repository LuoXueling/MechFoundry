from types import SimpleNamespace

import pytest

from src.agents_wrapper import (
    APIConfigurationError,
    APIRequestError,
    APIResponseError,
    GeminiAPI,
    GenerationRequest,
    ImageData,
)


class FakePart:
    @staticmethod
    def from_bytes(*, data, mime_type):
        return ("image", data, mime_type)


class FakeConfig:
    def __init__(self, **kwargs):
        self.values = kwargs


class FakeTypes:
    Part = FakePart
    GenerateContentConfig = FakeConfig


class FakeChat:
    def __init__(self, response="gemini answer", error=None):
        self.response = response
        self.error = error
        self.calls = []

    def send_message(self, contents):
        self.calls.append(contents)
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.response)


class FakeChats:
    def __init__(self, response="gemini answer", error=None):
        self.chat = FakeChat(response, error)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.chat


def make_api(response="gemini answer", error=None):
    chats = FakeChats(response=response, error=error)
    api = GeminiAPI(client=SimpleNamespace(chats=chats))
    api._types = FakeTypes
    return api, chats


def test_gemini_text_request_disables_afc():
    api, chats = make_api()
    result = api.generate_text(GenerationRequest(model="gemini-test", text="hi"))
    assert result == "gemini answer"
    assert chats.calls[0]["model"] == "gemini-test"
    assert chats.chat.calls[0] == "hi"
    assert chats.calls[0]["config"].values == {
        "automatic_function_calling": {"disable": True}
    }


def test_gemini_multimodal_request_and_config():
    api, chats = make_api()
    request = GenerationRequest(
        model="gemini-test",
        text="caption it",
        image=ImageData(b"image", "image/png"),
        system_prompt="be accurate",
        max_output_tokens=100,
        temperature=0.1,
        top_p=0.8,
        top_k=5,
        stop_sequences=("STOP",),
    )
    api.generate_text(request)
    call = chats.calls[0]
    assert chats.chat.calls[0] == [("image", b"image", "image/png"), "caption it"]
    assert call["config"].values == {
        "automatic_function_calling": {"disable": True},
        "system_instruction": "be accurate",
        "max_output_tokens": 100,
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 5,
        "stop_sequences": ["STOP"],
    }


def test_gemini_image_only_request():
    api, chats = make_api()
    api.generate_text(
        GenerationRequest(
            model="gemini-test", image=ImageData(b"image", "image/jpeg")
        )
    )
    assert chats.chat.calls[0] == [("image", b"image", "image/jpeg")]


def test_gemini_requests_native_json_mode():
    api, chats = make_api('{"ok":true}')
    api.generate_text(GenerationRequest(model="model", text="json", response_json=True))
    assert chats.calls[0]["config"].values["response_mime_type"] == "application/json"


def test_gemini_wraps_request_failure():
    api, _ = make_api(error=ConnectionError("offline"))
    with pytest.raises(APIRequestError, match="ConnectionError"):
        api.generate_text(GenerationRequest(model="model", text="hello"))


@pytest.mark.parametrize("response", [None, "", "  "])
def test_gemini_rejects_empty_response(response):
    api, _ = make_api(response=response)
    with pytest.raises(APIResponseError, match="blocked or empty"):
        api.generate_text(GenerationRequest(model="model", text="hello"))


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(APIConfigurationError, match="GEMINI_API_KEY"):
        GeminiAPI()
