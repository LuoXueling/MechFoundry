from types import SimpleNamespace

import pytest

from src.agents_wrapper import (
    APIConfigurationError,
    APIRequestError,
    APIResponseError,
    GenerationRequest,
    GroqAPI,
    ImageData,
)


class FakeCompletions:
    def __init__(self, response="groq answer", error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if self.response is ...:
            return SimpleNamespace(choices=[])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.response))]
        )


def make_api(response="groq answer", error=None):
    completions = FakeCompletions(response=response, error=error)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return GroqAPI(client=client), completions


def test_groq_text_request():
    api, completions = make_api()
    result = api.generate_text(
        GenerationRequest(
            model="groq-model",
            text="hello",
            system_prompt="be concise",
            max_output_tokens=100,
            temperature=0.2,
            top_p=0.8,
            stop_sequences=("STOP",),
        )
    )
    assert result == "groq answer"
    assert completions.calls == [
        {
            "model": "groq-model",
            "messages": [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
            ],
            "stream": False,
            "max_completion_tokens": 100,
            "temperature": 0.2,
            "top_p": 0.8,
            "stop": ["STOP"],
        }
    ]


def test_groq_image_is_encoded_as_data_url():
    api, completions = make_api()
    api.generate_text(
        GenerationRequest(
            model="vision-model",
            text="describe",
            image=ImageData(b"image bytes", "image/png"),
        )
    )
    content = completions.calls[0]["messages"][0]["content"]
    assert content == [
        {"type": "text", "text": "describe"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aW1hZ2UgYnl0ZXM="},
        },
    ]


def test_groq_rejects_unsupported_top_k():
    api, _ = make_api()
    with pytest.raises(APIConfigurationError, match="top_k"):
        api.generate_text(
            GenerationRequest(model="model", text="hello", top_k=10)
        )


def test_groq_requests_native_json_mode():
    api, completions = make_api('{"ok":true}')
    api.generate_text(GenerationRequest(model="model", text="json", response_json=True))
    assert completions.calls[0]["response_format"] == {"type": "json_object"}


def test_groq_wraps_request_failure_without_details():
    api, _ = make_api(error=ConnectionError("secret detail"))
    with pytest.raises(APIRequestError, match="ConnectionError") as caught:
        api.generate_text(GenerationRequest(model="model", text="hello"))
    assert "secret detail" not in str(caught.value)


@pytest.mark.parametrize("response", [None, "", "  "])
def test_groq_rejects_empty_response(response):
    api, _ = make_api(response=response)
    with pytest.raises(APIResponseError, match="no text"):
        api.generate_text(GenerationRequest(model="model", text="hello"))


def test_groq_rejects_malformed_response():
    api, _ = make_api(response=...)
    with pytest.raises(APIResponseError, match="unexpected"):
        api.generate_text(GenerationRequest(model="model", text="hello"))


def test_missing_groq_key_is_reported(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(APIConfigurationError, match="GROQ_API_KEY"):
        GroqAPI()
