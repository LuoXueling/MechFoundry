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


# ------------------------------------------------------------------- Codex + Hugging Face
# A stub `codex` (like experiments/agentic_batch/tests/stub_codex.py, but for the UI's command
# lines): it records its argv and the environment variables that matter, then plays a turn in the
# `codex exec --json` shapes measured on codex-cli 0.163.0, keyed by words in the prompt.
STUB_CODEX = r'''#!{python}
import json, os, sys, time
args = sys.argv[1:]
env = {k: os.environ.get(k) for k in ("CODEX_HOME", "FEM_OPERATOR_OUTPUT_ROOT", "FEM_OPERATOR_PROVIDER",
                                      "FEM_OPERATOR_COMPUTE_ENV", "CODEX_THREAD_ID", "CLAUDE_STUB_LEAK")}
env["HF_TOKEN_ok"] = os.environ.get("HF_TOKEN") == "hf_test"
env["stdin"] = sys.stdin.read()
with open(os.environ["STUB_RECORD"], "a") as fh:
    fh.write(json.dumps({"args": args, "env": env, "cwd": os.getcwd()}) + "\n")
prompt = args[-1]
resumed = args[1] == "resume"
thread = args[-2] if resumed else "01a1eb4d-0000-7000-8000-00000000abcd"
def emit(rec):
    print(json.dumps(rec), flush=True)
emit({"type": "thread.started", "thread_id": thread})
emit({"type": "item.completed", "item": {"id": "item_0", "type": "error",
       "message": "Model metadata for `x` not found. Defaulting to fallback metadata"}})
emit({"type": "turn.started"})
if "hang" in prompt:
    time.sleep(600)
if "fail" in prompt:
    emit({"type": "error", "message": "Reconnecting... 1/5 (402)"})
    emit({"type": "turn.failed", "error": {"message": "402 Purchase pre-paid credits"}})
    sys.exit(1)
emit({"type": "item.completed", "item": {"id": "item_1", "type": "agent_message", "text": "Starting."}})
cmd = "python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py start --instruction x"
emit({"type": "item.started", "item": {"id": "item_2", "type": "command_execution",
       "command": "/bin/bash -lc '" + cmd + "'", "aggregated_output": "", "exit_code": None,
       "status": "in_progress"}})
emit({"type": "item.completed", "item": {"id": "item_2", "type": "command_execution",
       "command": "/bin/bash -lc '" + cmd + "'", "exit_code": 0, "status": "completed",
       "aggregated_output": "session_id: abcdef012345\n" + json.dumps({"ok": True, "session_id": "abcdef012345"})}})
day = os.path.join(os.environ["CODEX_HOME"], "sessions", "2026", "11", "17")
os.makedirs(day, exist_ok=True)
with open(os.path.join(day, "rollout-2026-11-17T15-00-00-" + thread + ".jsonl"), "a") as fh:
    fh.write(json.dumps({"type": "response_item", "payload": {"type": "function_call", "name": "view_image",
             "arguments": json.dumps({"path": os.environ["STUB_PICTURE"]}), "call_id": "call-v" + str(len(prompt))}}) + "\n")
plan = os.environ["FEM_OPERATOR_OUTPUT_ROOT"] + "/abcdef012345/operator/plan.json"
emit({"type": "item.completed", "item": {"id": "item_3", "type": "file_change",
       "changes": [{"path": plan, "kind": "add"}], "status": "completed"}})
emit({"type": "item.completed", "item": {"id": "item_4", "type": "agent_message", "text": "Done: 2 GPUs."}})
emit({"type": "turn.completed", "usage": {"input_tokens": 100000, "cached_input_tokens": 80000,
       "cache_write_input_tokens": 0, "output_tokens": 3000, "reasoning_output_tokens": 1000}})
'''

FAKE_RELAY = r'''#!{python}
import sys, time
a = sys.argv
open(a[a.index("--port-file") + 1], "w").write("54321")
time.sleep(600)
'''


@pytest.fixture
def codex_hub(tmp_path, monkeypatch):
    root = tmp_path / "fem_workflow"
    (root / "abcdef012345" / "operator").mkdir(parents=True)
    picture = root / "abcdef012345" / "views.png"
    picture.write_bytes(b"\x89PNG")
    stub = tmp_path / "codex"
    stub.write_text(STUB_CODEX.replace("{python}", sys.executable))
    stub.chmod(0o755)
    relay = tmp_path / "fake_relay.py"
    relay.write_text(FAKE_RELAY.replace("{python}", sys.executable))
    monkeypatch.setattr(A, "HF_RELAY", relay)
    monkeypatch.setenv("CODEX_CLI", str(stub))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("STUB_RECORD", str(tmp_path / "stub_calls.jsonl"))
    monkeypatch.setenv("STUB_PICTURE", str(picture))
    monkeypatch.setenv("CODEX_THREAD_ID", "parent-thread")      # must not reach the child
    monkeypatch.setenv("CLAUDE_STUB_LEAK", "1")
    hub = A.AgentHub(root)
    yield hub
    hub.shutdown()


def create_codex(hub, effort="medium"):
    return hub.create(provider="deepseek", model="deepseek-flash", vision_provider="groq",
                      vision_model="qwen", compute_env="mechfoundry-warp117", operator="codex",
                      codex_model="deepseek-v4.1-flash", effort=effort)


def stub_calls(tmp_path):
    return [json.loads(l) for l in (tmp_path / "stub_calls.jsonl").read_text().splitlines()]


def test_codex_status(tmp_path, monkeypatch):
    cli = tmp_path / "codex"
    cli.write_text("#!/bin/sh\n")
    cli.chmod(0o755)
    monkeypatch.setenv("CODEX_CLI", str(cli))
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    st = A.codex_status()
    assert st.ready and st.cli == str(cli) and "HF_TOKEN" in st.summary
    # No HF_TOKEN: the `hf auth login` token file.
    monkeypatch.delenv("HF_TOKEN")
    token_file = tmp_path / "hf" / "token"
    token_file.parent.mkdir()
    token_file.write_text("hf_file\n")
    monkeypatch.setenv("HF_HOME", str(token_file.parent))
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    assert A.hf_token() == ("hf_file", str(token_file)) and A.codex_status().ready
    token_file.unlink()
    st = A.codex_status()
    assert not st.ready and "Hugging Face token" in st.summary
    monkeypatch.setenv("CODEX_CLI", "/nonexistent/codex")
    monkeypatch.setenv("PATH", "/nonexistent")
    st = A.codex_status()
    assert not st.ready and "codex" in st.summary and "Hugging Face token" in st.summary
    assert A.operator_status("codex").summary == st.summary


def test_codex_meta_and_label(codex_hub):
    conv = create_codex(codex_hub)
    assert isinstance(conv, A.CodexConversation) and conv.operator == "codex"
    meta = conv.meta
    assert meta["operator"] == "codex" and meta["codex_model"] == "deepseek-v4.1-flash"
    assert meta["codex_route"] == "deepinfra" and meta["effort"] == "medium" and meta["claude_model"] == ""
    assert A.operator_label(meta) == "Codex · DeepSeek V4.1 Flash (Hugging Face, deepinfra) · medium effort"
    # A restarted server reads the operator back; a conversation from before the choice is Claude's.
    again = A.AgentHub(codex_hub.output_root)
    assert isinstance(again.get(conv.id), A.CodexConversation)
    old = again.create(provider="p", model="m", vision_provider="p", vision_model="v", compute_env="e")
    (old.dir / "meta.json").write_text(json.dumps({k: v for k, v in old.meta.items() if k != "operator"}))
    assert isinstance(A.AgentHub(codex_hub.output_root).get(old.id), A.AgentConversation)
    assert A.operator_label({"claude_model": "claude-opus-5-5", "effort": "high"}) == \
        "Claude Code · claude-opus-5-5 · high effort"
    with pytest.raises(ValueError):
        codex_hub.create(provider="p", model="m", vision_provider="p", vision_model="v",
                         compute_env="e", operator="gpt")


def test_codex_home_through_the_relay(codex_hub):
    import tomllib
    url = codex_hub.call(codex_hub.hf_relay()).result(timeout=15)
    assert url == "http://127.0.0.1:54321/v1"
    relay = codex_hub.relay
    assert codex_hub.call(codex_hub.hf_relay()).result(timeout=5) == url and codex_hub.relay is relay
    conv = create_codex(codex_hub, effort="high")
    model = codex_hub.call(conv.write_home(url)).result(timeout=30)
    assert model["model"] == "deepseek-ai/DeepSeek-V4.1-Flash-1017:deepinfra"
    cfg = tomllib.loads((conv.home / "config.toml").read_text())
    assert cfg["model_provider"] == "hf" and cfg["model_providers"]["hf"]["base_url"] == url
    assert cfg["model_providers"]["hf"]["env_key"] == "HF_TOKEN" and "hf_test" not in (conv.home / "config.toml").read_text()
    assert cfg["approval_policy"] == "never" and cfg["sandbox_mode"] == "workspace-write"
    assert cfg["model_reasoning_effort"] == "high" and cfg["features"]["enable_request_compression"] is False
    notes = cfg["developer_instructions"]
    # The path note, then codex_tools.txt, then the UI's note (which wins where they differ); not
    # the unattended batch's codex_note.txt ("nobody answers in this run").
    assert notes.index("Environment note") < notes.index("Tool note:") < notes.index("UI note:")
    assert "Batch note:" not in notes and "write_stdin" in notes
    assert str(codex_hub.output_root) in notes and "deepseek/deepseek-flash" in notes
    assert "drive.py" in (conv.home / "rules" / "operator.rules").read_text()
    meta = conv.meta
    assert meta["codex_model_id"] == model["model"] and meta["codex_price"]["output"] == 1.2
    # The route chosen on the card stays; the service ab_helper resolved it to is kept beside it.
    assert meta["codex_provider"] == "hf" and meta["codex_service"] == "hf"
    codex_hub.shutdown()
    assert relay.wait(5) is not None and codex_hub.relay is None


def test_codex_turn_records_events(codex_hub, tmp_path):
    conv = create_codex(codex_hub)
    assert conv.send("how many GPUs?", [])
    wait_idle(conv, timeout=30)
    events = conv.snapshot()["events"]
    errors = [e for e in events if e["type"] == "error"]
    assert not errors, errors
    types = [e["type"] for e in events]
    assert types[0] == "user" and types[-1] == "result" and "notice" in types
    uses = {e["name"]: e for e in events if e["type"] == "tool_use"}
    # Codex's file_change (apply_patch) shows as Write when it only adds files (operators.py).
    assert set(uses) == {"Bash", "Write", "view_image"}
    # The command as typed, not Codex's /bin/bash -lc '...' wrapper.
    assert uses["Bash"]["input"]["command"].startswith("python3 -u src/agents_wrapper")
    results = {e["tool_use_id"]: e for e in events if e["type"] == "tool_result"}
    assert set(results) == {e["id"] for e in uses.values()}
    assert results[uses["view_image"]["id"]]["image"].endswith("views.png")
    assert [e["text"] for e in events if e["type"] == "assistant_text"] == ["Starting.", "Done: 2 GPUs."]
    result = events[-1]
    assert result["is_error"] is False and result["cost_usd"] == pytest.approx(
        (20000 * 0.3 + 80000 * 0.006 + 3000 * 1.2) / 1e6)
    meta = conv.meta
    assert meta["codex_thread_id"] == "01a1eb4d-0000-7000-8000-00000000abcd"
    assert meta["sessions"] == ["abcdef012345"] and meta["current_session"] == "abcdef012345"
    assert codex_hub.owned_sessions() == {"abcdef012345": conv.id}
    call = stub_calls(tmp_path)[0]
    assert call["args"][:2] == ["exec", "--json"] and call["args"][2:4] == ["-C", str(A.REPO_ROOT)]
    assert call["args"][-2:] == ["--", "how many GPUs?"]
    env = call["env"]
    assert env["CODEX_HOME"] == str(conv.home) and env["HF_TOKEN_ok"] and env["stdin"] == ""
    assert env["FEM_OPERATOR_OUTPUT_ROOT"] == str(codex_hub.output_root)
    assert env["FEM_OPERATOR_PROVIDER"] == "deepseek" and env["FEM_OPERATOR_COMPUTE_ENV"] == "mechfoundry-warp117"
    assert env["CODEX_THREAD_ID"] is None and env["CLAUDE_STUB_LEAK"] is None
    assert call["cwd"] == str(A.REPO_ROOT)
    assert "hf_test" not in (conv.dir / "chat.jsonl").read_text() + (conv.dir / "meta.json").read_text()


def test_codex_resume_with_an_image(codex_hub, tmp_path):
    conv = create_codex(codex_hub)
    conv.send("first", [])
    wait_idle(conv, timeout=30)
    pic = conv.save_upload("sketch.png", b"\x89PNG")
    mesh = conv.save_upload("part.stl", b"solid")
    cmd = conv.command("again", [pic, mesh])
    thread = conv.meta["codex_thread_id"]
    assert cmd[1:4] == ["exec", "resume", "--json"] and "-C" not in cmd
    assert cmd[cmd.index("-i") + 1] == str(pic) and cmd.count("-i") == 1      # pictures only
    assert cmd[-3:-1] == ["--", thread] and str(mesh) in cmd[-1] and str(pic) in cmd[-1]
    roots = json.loads(cmd[cmd.index("-c") + 1].split("=", 1)[1])
    assert str(codex_hub.output_root) in roots
    conv.send("again", [pic])
    wait_idle(conv, timeout=30)
    second = stub_calls(tmp_path)[-1]["args"]
    assert second[:3] == ["exec", "resume", "--json"] and second[-2] == thread
    # Item ids restart every turn; the page's ids do not.
    uses = [e["id"] for e in conv.events if e["type"] == "tool_use"]
    assert len(uses) == len(set(uses)) == 6
    assert sum(e["type"] == "result" for e in conv.events) == 2


def test_codex_interrupt_terminates_the_process(codex_hub):
    conv = create_codex(codex_hub)
    conv.send("hang here", [])
    t0 = time.time()
    while conv.proc is None and time.time() - t0 < 30:
        time.sleep(0.05)
    assert conv.proc is not None
    time.sleep(0.3)
    conv.interrupt()
    wait_idle(conv, timeout=20)
    result = conv.events[-1]
    assert result["type"] == "result" and result["reason"] == "aborted_by_user" and not result["is_error"]
    assert not [e for e in conv.events if e["type"] == "error"]


def test_codex_failed_turn(codex_hub):
    conv = create_codex(codex_hub)
    conv.send("fail please", [])
    wait_idle(conv, timeout=30)
    errors = [e["message"] for e in conv.events if e["type"] == "error"]
    assert errors == ["Codex: 402 Purchase pre-paid credits"]
    assert conv.events[-1]["type"] == "result" and conv.events[-1]["is_error"]


def test_codex_without_token_says_so(codex_hub, monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "missing"))
    conv = create_codex(codex_hub)
    conv.send("hi", [])
    wait_idle(conv, timeout=10)
    assert "Hugging Face token" in conv.events[-1]["message"]
    assert not (tmp_path / "stub_calls.jsonl").exists()


def test_codex_vendor_route(codex_hub, monkeypatch):
    import tomllib
    """DeepSeek's own API: Codex talks to it directly (no relay), with DEEPSEEK_API_KEY; the
    conversation keeps its route across turns (ab_helper names the service "deepseek")."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk_test")
    conv = codex_hub.create(provider="deepseek", model="deepseek-flash", vision_provider="groq",
                            vision_model="qwen", compute_env="mechfoundry-warp117", operator="codex",
                            codex_model="deepseek-v4.1-flash", codex_provider="vendor", effort="medium")
    for _ in range(2):                                   # each turn rewrites the home
        codex_hub.call(conv.write_home("")).result(timeout=30)
    cfg = tomllib.loads((conv.home / "config.toml").read_text())
    prov = cfg["model_providers"][cfg["model_provider"]]
    assert prov["base_url"] == "https://api.deepseek.com" and prov["env_key"] == "DEEPSEEK_API_KEY"
    assert "sk_test" not in (conv.home / "config.toml").read_text() and codex_hub.relay is None
    meta = conv.meta
    assert meta["codex_provider"] == "vendor" and meta["codex_service"] == "deepseek"
    assert conv.provider_key() == ("DEEPSEEK_API_KEY", "sk_test")
    assert A.operator_label(meta) == "Codex · DeepSeek V4.1 Flash (own API) · medium effort"
