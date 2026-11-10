"""Agent sessions (src/ui/agent_session.py) with a fake Claude client: no API calls, no CLI."""
import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

sdk = pytest.importorskip("claude_agent_sdk")
from claude_agent_sdk import (AssistantMessage, ResultMessage, StreamEvent, SystemMessage,  # noqa: E402
                              TextBlock, ToolResultBlock, ToolUseBlock, UserMessage)

from src.ui import agent_session as A  # noqa: E402


class FakeClient:
    """Plays one turn: a streamed sentence, a drive.py start that prints a session id, a Read
    of a picture, and the result. With `ask` it asks permission for a command first."""
    last = None

    def __init__(self, options):
        self.options = options
        self.sent = []
        self.interrupted = False
        FakeClient.last = self

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def interrupt(self):
        self.interrupted = True

    async def query(self, gen):
        async for m in gen:
            self.sent.append(m)

    async def receive_response(self):
        yield SystemMessage(subtype="init", data={"session_id": "claude-1"})
        if self.ask:
            ok = await self.options.can_use_tool("Bash", {"command": "touch x"}, None)
            yield AssistantMessage(content=[TextBlock(text=f"permission {ok.behavior}")], model="m")
        yield StreamEvent(uuid="u", session_id="claude-1",
                          event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Start"}})
        yield AssistantMessage(content=[
            TextBlock(text="Starting."),
            ToolUseBlock(id="t1", name="Bash", input={"command": "python3 -u src/agents_wrapper/fem_workflow/"
                                                                "operator_skill/scripts/drive.py start --instruction x"}),
        ], model="m")
        yield UserMessage(content=[ToolResultBlock(tool_use_id="t1", content='{\n"ok":true,\n"session_id":"abcdef012345"\n}')])
        yield AssistantMessage(content=[ToolUseBlock(id="t2", name="Read", input={"file_path": self.picture})], model="m")
        yield UserMessage(content=[ToolResultBlock(tool_use_id="t2", content=[{"type": "image"}])])
        yield ResultMessage(subtype="success", duration_ms=5, duration_api_ms=4, is_error=False,
                            num_turns=1, session_id="claude-1", total_cost_usd=0.01)


def wait_idle(conv, timeout=10):
    t0 = time.time()
    while conv.busy and time.time() - t0 < timeout:
        time.sleep(0.02)
    assert not conv.busy


@pytest.fixture
def hub(tmp_path):
    root = tmp_path / "fem_workflow"
    (root / "abcdef012345").mkdir(parents=True)
    FakeClient.ask = False
    FakeClient.picture = str(root / "abcdef012345" / "views.png")
    return A.AgentHub(root, client_factory=FakeClient)


def create(hub):
    return hub.create(provider="deepseek", model="deepseek-flash", vision_provider="groq",
                      vision_model="qwen", compute_env="mechfoundry-warp117",
                      claude_model="claude-sonnet-5-5", effort="medium")


def test_turn_records_events_and_follows_the_session(hub):
    conv = create(hub)
    assert conv.send("simulate a hook", [])
    wait_idle(conv)
    types = [e["type"] for e in conv.snapshot()["events"]]
    assert types[0] == "user" and "tool_use" in types and types[-1] == "result"
    meta = conv.meta
    assert meta["claude_session_id"] == "claude-1"
    assert meta["sessions"] == ["abcdef012345"] and meta["current_session"] == "abcdef012345"
    assert hub.owned_sessions() == {"abcdef012345": conv.id}
    read = [e for e in conv.events if e["type"] == "tool_result" and e.get("image")]
    assert read and read[0]["image"].endswith("views.png")
    # Deltas are drawn live but not kept.
    saved = [json.loads(l)["type"] for l in (conv.dir / "chat.jsonl").read_text().splitlines()]
    assert "delta" not in saved


def test_options_carry_the_setup_choices(hub):
    conv = create(hub)
    opts = conv.options()
    assert opts.env["FEM_OPERATOR_OUTPUT_ROOT"] == str(hub.output_root)
    assert opts.env["FEM_OPERATOR_PROVIDER"] == "deepseek"
    assert opts.env["FEM_OPERATOR_VISION_PROVIDER"] == "groq"
    assert opts.env["FEM_OPERATOR_COMPUTE_ENV"] == "mechfoundry-warp117"
    assert opts.model == "claude-sonnet-5-5" and opts.cwd == str(A.REPO_ROOT)
    assert any("drive.py" in a for a in opts.allowed_tools) and "Bash(cat *)" in opts.disallowed_tools
    assert opts.max_buffer_size >= 32 * 1024 * 1024
    assert opts.setting_sources == []


def test_resume_and_busy(hub):
    conv = create(hub)
    conv.update_meta(claude_session_id="earlier")
    assert conv.options().resume == "earlier"
    conv.send("one", [])
    wait_idle(conv)
    # A second hub (a restarted server) reads the conversation back from disk.
    again = A.AgentHub(hub.output_root, client_factory=FakeClient).get(conv.id)
    assert [e["type"] for e in again.events] == [e["type"] for e in conv.events]


def test_permission_relay(hub):
    FakeClient.ask = True
    conv = create(hub)
    conv.send("go", [])
    t0 = time.time()
    while not conv.snapshot()["pending"] and time.time() - t0 < 5:
        time.sleep(0.02)
    pending = conv.snapshot()["pending"]
    assert pending and pending[0]["tool"] == "Bash"
    conv.answer_permission(pending[0]["id"], False)
    wait_idle(conv)
    texts = [e.get("text") for e in conv.events if e["type"] == "assistant_text"]
    assert "permission deny" in texts
    assert any(e["type"] == "permission_done" and e["allowed"] is False for e in conv.events)


def test_attachments_become_image_blocks(hub, tmp_path):
    conv = create(hub)
    pic = conv.save_upload("../../sketch.png", b"\x89PNG")
    assert pic.parent == conv.dir / "uploads" and pic.name == "sketch.png"
    conv.send("this one", [pic])
    wait_idle(conv)
    content = FakeClient.last.sent[0]["message"]["content"]
    assert content[0]["type"] == "image" and "sketch.png" in content[-1]["text"]


def test_session_tracking_ignores_unknown_ids(hub):
    conv = create(hub)
    conv._track('drive.py run 111111111111 --plan x')        # not a folder in the root
    assert conv.meta["sessions"] == []
    conv._track("python3 -u src/x/operator_skill/scripts/drive.py run abcdef012345 --plan p")
    assert conv.meta["current_session"] == "abcdef012345"


def test_status_reports_a_missing_cli(monkeypatch):
    monkeypatch.setenv("CLAUDE_CLI", "/nonexistent/claude")
    monkeypatch.setenv("PATH", "/nonexistent")
    st = A.agent_status()
    assert not st.ready and "claude" in st.summary


def test_status_reports_a_missing_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    st = A.agent_status()
    assert not st.ready and "SDK" in st.summary
