"""src/ui/operators.py: the operator list, the Codex login sharing, an OpenAI CODEX_HOME, and how
a Codex agent's commands read. No network, no model call."""
import json
import shlex
import tomllib

import pytest

from src.ui import operators as O


def stub_codex(tmp_path, catalog):
    cli = tmp_path / "codex"
    out = tmp_path / "catalog.json"
    out.write_text(json.dumps(catalog))
    cli.write_text(f"#!/bin/sh\n[ \"$1 $2\" = \"debug models\" ] && cat {out}\n")
    cli.chmod(0o755)
    return cli


CATALOG = {"models": [
    {"slug": "gpt-a", "display_name": "GPT-A", "visibility": "list", "input_modalities": ["text", "image"],
     "tool_mode": "code_mode_only",
     "supported_reasoning_levels": [{"effort": "low"}, {"effort": "max"}, {"effort": "ultra"}],
     "default_reasoning_level": "low"},
    {"slug": "gpt-hidden", "visibility": "hide", "input_modalities": ["text", "image"]},
    {"slug": "gpt-text", "visibility": "list", "input_modalities": ["text"]},
]}


@pytest.fixture
def login(tmp_path, monkeypatch):
    home = tmp_path / "dotcodex"
    home.mkdir()
    (home / "auth.json").write_text(json.dumps({"tokens": {"refresh_token": "OLD"}, "last_refresh": "2026-11-01"}))
    monkeypatch.setenv("CODEX_LOGIN_HOME", str(home))
    monkeypatch.setenv("CODEX_CLI", str(stub_codex(tmp_path, CATALOG)))
    O._OPENAI_CACHE.clear()
    yield home / "auth.json"
    O._OPENAI_CACHE.clear()


def test_openai_models_from_the_catalog(login):
    assert O.openai_models() == [{"slug": "gpt-a", "label": "GPT-A", "efforts": ["low", "max"],
                                  "default_effort": "low", "tool_mode": "code_mode_only"}]   # listed, images; no "ultra"


def test_operators(login, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    ops = {o["id"]: o for o in O.operators()}
    assert ops["claude-opus-5-5"]["group"] == O.GROUPS["claude"]
    assert ops["codex:gpt-a@openai"]["group"] == O.GROUPS["openai"] and ops["codex:gpt-a@openai"]["available"]
    hf = [o for o in ops.values() if o["provider"] == "hf"]
    assert hf and all(o["available"] and o["efforts"] == list(O.OPEN_EFFORTS) for o in hf)
    assert O.parse("codex:kimi-k3.1-pro@vendor") == {"agent": "codex", "model": "kimi-k3.1-pro", "provider": "vendor"}
    assert O.parse("claude-opus-5-5")["agent"] == "claude"
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HF_TOKEN_PATH", "/nonexistent")
    assert all(not o["available"] and o["unavailable"] == "no HF_TOKEN"
               for o in O.operators() if o["provider"] == "hf")


def test_login_sync(login, tmp_path):
    home = tmp_path / "private"
    O.sync_login(home, into_home=True)
    mine = home / "auth.json"
    assert json.loads(mine.read_text())["tokens"]["refresh_token"] == "OLD" and oct(mine.stat().st_mode)[-3:] == "600"
    O.sync_login(home, into_home=False)                     # not refreshed: nothing copied back
    assert json.loads(login.read_text())["tokens"]["refresh_token"] == "OLD"
    mine.write_text(json.dumps({"tokens": {"refresh_token": "NEW"}, "last_refresh": "2026-11-17"}))
    O.sync_login(home, into_home=False)                     # refreshed during the turn
    assert json.loads(login.read_text())["tokens"]["refresh_token"] == "NEW"


def test_openai_home(login, tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("the note")
    home = tmp_path / "codex_home"
    model = O.write_openai_home(home, "gpt-a", "high", [note])     # high is not a level of gpt-a
    cfg = tomllib.loads((home / "config.toml").read_text())
    assert cfg["model_provider"] == "openai" and cfg["model"] == "gpt-a" and cfg["model_reasoning_effort"] == "low"
    assert cfg["approval_policy"] == "never" and cfg["developer_instructions"] == "the note"
    assert cfg["features"]["multi_agent"] is False and "model_providers" not in cfg
    assert "code_mode_host" not in cfg["features"]        # code_mode_only: its tools need the host
    assert "drive.py" in (home / "rules" / "operator.rules").read_text()
    assert (home / "auth.json").exists() and model["provider"] == "openai" and model["price"] is None


def test_codex_commands():
    assert O.unwrap("/bin/bash -lc 'python3 -u x/drive.py run S'") == "python3 -u x/drive.py run S"
    assert O.classify("/bin/bash -lc \"sed -n '1,40p' /a/SKILL.md\"") == ("Read", {"file_path": "/a/SKILL.md", "lines": "1,40"})
    assert O.classify("/bin/bash -lc 'cat /a/b.md'") == ("Read", {"file_path": "/a/b.md"})
    assert O.classify("/bin/bash -lc 'ls -R output'")[0] == "Glob"
    assert O.classify("/bin/bash -lc 'grep -rn foo /a'") == ("Grep", {"pattern": "foo", "path": "/a"})
    assert O.classify("/bin/bash -lc 'cat a | head'")[0] == "Bash"
    assert O.classify("/bin/bash -lc 'python3 -u s/drive.py status S'") == ("Bash", {"command": "python3 -u s/drive.py status S"})
    # apply_patch in the shell, as the tool note has models without the apply_patch tool run it
    add = "apply_patch <<'PATCH'\n*** Begin Patch\n*** Add File: /s/operator/brief.md\n+x\n*** End Patch\nPATCH"
    name, inp = O.classify("/bin/bash -lc " + shlex.quote(add))
    assert name == "Write" and inp["file_path"] == "/s/operator/brief.md" and inp["changes"] == [
        {"path": "/s/operator/brief.md", "kind": "add"}]
    both = add.replace("*** End Patch", "*** Update File: /s/operator/plan.json\n@@\n-a\n+b\n*** End Patch")
    name, inp = O.classify(both)
    assert name == "Edit" and inp["file_path"] == "/s/operator/brief.md, /s/operator/plan.json"
    assert O.classify("apply_patch --help")[0] == "Bash"           # names no file: as typed
    raw = '{"success":false,"error":[{"issues":[{"message":"Required"}],"message":"Invalid input"}]}'
    assert O.short_error(raw) == "Invalid input"
    assert O.short_error("402 Purchase pre-paid credits") == "402 Purchase pre-paid credits"


def test_claude_models_from_claude_code(monkeypatch):
    """The current model of each family, as Claude Code's initialize reply lists them; the built-in
    list when Claude Code cannot be asked."""
    catalog = [
        {"value": "default", "resolvedModel": "claude-opus-9", "displayName": "Default (recommended)"},
        {"value": "opus", "resolvedModel": "claude-opus-9", "displayName": "Opus 9", "supportsEffort": True,
         "supportedEffortLevels": ["low", "medium", "high", "max"]},
        {"value": "haiku", "resolvedModel": "claude-haiku-9", "displayName": "Haiku 9"},
        {"value": "claude-opus-4-6", "resolvedModel": "claude-opus-4-6", "displayName": "Opus 4.6"},
    ]
    monkeypatch.setattr(O, "claude_cli", lambda: "/bin/true")
    monkeypatch.setattr(O, "_claude_catalog", lambda: catalog)
    O._CLAUDE_CACHE.clear()
    assert O.claude_models() == [
        {"id": "claude-opus-9", "label": "Claude Opus 9", "efforts": ["low", "medium", "high", "max"],
         "default_effort": "high"},
        {"id": "claude-haiku-9", "label": "Claude Haiku 9", "efforts": [], "default_effort": None}]
    O._CLAUDE_CACHE.clear()
    monkeypatch.setattr(O, "_claude_catalog", lambda: [])
    assert [m["id"] for m in O.claude_models()] == [m for m, _ in O.CLAUDE_MODELS]
    O._CLAUDE_CACHE.clear()
