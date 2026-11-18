"""Agent-controlled sessions: an AI operator that drives the workflow for the user.

The UI's setup card picks the workflow's text and vision models and the compute environment, as
for any conversation, and the operator:

* Claude Code (the locally installed ``claude`` CLI, on the user's own login, through the Claude
  Agent SDK; ``AgentConversation``), or
* Codex (``CodexConversation``) on an open-weight model through Hugging Face or its company's own
  API, or on an OpenAI model with the user's own Codex login: the ``codex`` CLI
  run headless, ``codex exec --json``, once per user message, as the agentic batch's
  ``run_batch.sh --agent codex`` runs it. Its pieces are reused from experiments/agentic_batch/
  rather than copied: the models (``codex_models.json``), the private CODEX_HOME (``ab_helper.py
  codex-home``), the tool-name note (``codex_tools.txt``; not the batch's own ``codex_note.txt``,
  written for unattended runs) and the Responses-to-Chat translator
  Hugging Face needs (``hf_relay.py``; its docstring says why). Which operators exist here, their
  keys, and how Codex's commands read are in ``operators.py`` (shared with other front ends).

"Start agent session" then hands the work to the operator, which follows the operator skill
(``src/agents_wrapper/fem_workflow/operator_skill/SKILL.md``) and drives ``drive.py``. The page
shows the session the agent is working on, read-only, and refreshes as its files change; the user
can send further prompts, stop a reply, and (Claude only) approve or deny any action outside the
allow list. Codex asks nothing: it runs with approval "never", its own commands in its sandbox,
and only the runner's two command forms let out of it by its rules.

Streamlit-free, so it can be tested: one background thread runs an asyncio loop that owns every
SDK client and Codex process; the page talks to it through thread-safe calls and reads plain
snapshots.

Storage: ``<output root>_agent/<conversation id>/``
  meta.json   operator, providers, models, compute env, Claude session or Codex thread id, the
              workflow sessions touched
  chat.jsonl  the conversation as events (text deltas are not kept)
  codex_home/ (Codex) the conversation's own CODEX_HOME: config.toml, rules/, sessions/ (rollout)
  notes/      (Codex) the developer notes config.toml was written from
And ``<output root>_agent/.hf_relay/`` the relay's port file and request log (no bodies, no headers).

The agent's workflow sessions live in the UI's own output root (``FEM_OPERATOR_OUTPUT_ROOT``), so
they are listed in the sidebar like any other; ``drive.py start`` takes the setup card's choices
from ``FEM_OPERATOR_PROVIDER`` and its siblings.
"""
from __future__ import annotations

import asyncio
import atexit
import base64
import importlib.util
import json
import mimetypes
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.ui import operators as OPS

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL = "src/agents_wrapper/fem_workflow/operator_skill"
IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
CLAUDE_MODELS = tuple(m for m, _ in OPS.CLAUDE_MODELS)
EFFORTS = OPS.CLAUDE_EFFORTS
# The shell's file readers and stdin Python, refused outright as in the agentic batch
# (experiments/agentic_batch/ab_common.sh, denied_tools): the skill reads with the Read tool.
SHELL_READERS = ("cat head tail less more grep egrep fgrep rg find sed awk ls tree file stat wc od xxd "
                 "strings sort uniq cut diff cmp nl tac realpath readlink du git rm sudo conda pip").split()
SESSION_IN_COMMAND = re.compile(r"(?:drive|inspect_session)\.py\s+(?:--\S+\s+\S+\s+)*\w[\w-]*\s+([0-9a-f]{12})\b")
SESSION_IN_RESULT = re.compile(r'"session_id"\s*:\s*"([0-9a-f]{12})"')
SESSION_LINE = re.compile(r"session_id: ([0-9a-f]{12})")

OPERATORS = {"claude": "Claude Code", "codex": "Codex"}
# The agentic batch's Codex pieces, used where they are (never copied, so a fix there reaches
# the UI too): ab_helper.py writes a run's CODEX_HOME, codex_tools.txt maps the skill's Claude
# tool names to Codex's, codex_models.json lists the models, hf_relay.py translates for HF.
AB_DIR, AB_HELPER, HF_RELAY, CODEX_TOOLS = OPS.AB_DIR, OPS.AB_HELPER, OPS.HF_RELAY, OPS.CODEX_TOOLS
CODEX_NOTE, CODEX_MODELS_FILE, HF_UPSTREAM = OPS.CODEX_NOTE, OPS.CODEX_MODELS_FILE, OPS.HF_UPSTREAM
CODEX_EFFORTS = OPS.OPEN_EFFORTS
# One `codex exec --json` line can carry a whole command's output (a `cat SKILL.md` is ~40 kB);
# asyncio's 64 kB line default would end the turn with a LimitOverrunError.
CODEX_LINE_LIMIT = 64 * 1024 * 1024
# Codex's own startup warning for a model it has no metadata for (every HF model); not news to
# the user, so it is kept in chat.jsonl but not shown.
CODEX_QUIET_ERRORS = ("Model metadata for",)


# ------------------------------------------------------------------------------- availability
@dataclass(frozen=True)
class AgentStatus:
    ready: bool
    summary: str
    cli: str | None = None


def agent_status() -> AgentStatus:
    """Whether agent sessions can be offered: the SDK in this env and a ``claude`` CLI to run.

    ``CLAUDE_CLI`` names the CLI when it is not on the PATH. The login itself is not probed (that
    would cost a call); a missing login is reported by the first message."""
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        return AgentStatus(False, "Claude Agent SDK is not installed in this environment "
                                  "(pip install claude-agent-sdk)")
    cli = os.environ.get("CLAUDE_CLI") or shutil.which("claude")
    if not cli or not Path(cli).exists():
        return AgentStatus(False, "Claude Code (the `claude` command) is not installed or not on the PATH")
    return AgentStatus(True, f"Claude Code at {cli}", cli)


def hf_token() -> tuple[str, str]:
    """The Hugging Face token and where it came from (operators.hf_token). The value goes only into
    the Codex process's environment, never into a file or a log; Codex's default
    shell_environment_policy keeps *TOKEN* variables out of the agent's own commands."""
    return OPS.hf_token()


def codex_status() -> AgentStatus:
    """Whether Codex sessions can be offered: a ``codex`` CLI (``CODEX_CLI`` or the PATH), a
    Hugging Face token, and the agentic batch's Codex files. Nothing is probed over the network
    (that would cost a call); a refused token or empty credit is reported by the first message."""
    missing = []
    cli = os.environ.get("CODEX_CLI") or shutil.which("codex")
    if not cli or not Path(cli).exists():
        missing.append("Codex (the `codex` command) is not installed or not on the PATH")
        cli = None
    token, source = hf_token()
    if not token:
        missing.append("no Hugging Face token (set HF_TOKEN or run `hf auth login`)")
    absent = [p.name for p in (AB_HELPER, HF_RELAY, CODEX_TOOLS, CODEX_MODELS_FILE) if not p.exists()]
    if absent:
        missing.append(f"experiments/agentic_batch lacks {', '.join(absent)}")
    if missing:
        return AgentStatus(False, "; ".join(missing), cli)
    return AgentStatus(True, f"Codex at {cli}, Hugging Face token from {source}", cli)


def operator_status(operator: str) -> AgentStatus:
    return codex_status() if operator == "codex" else agent_status()


def ab_helper():
    """experiments/agentic_batch/ab_helper.py as a module (operators.ab_helper)."""
    return OPS.ab_helper()


def codex_models() -> dict[str, dict]:
    """codex_models.json's models: key -> {label, effort, ...}; {} when the file is absent."""
    return OPS.codex_models()


def codex_operators() -> list[dict]:
    """The Codex operators (open-weight through Hugging Face or a company API, and OpenAI models on
    the user's Codex login), each with whether it can run here and why not (operators.operators)."""
    return [o for o in OPS.operators() if o["agent"] == "codex"]


def operator_label(meta: dict) -> str:
    """Which operator and model a conversation runs, for the agent view's heading."""
    effort = f" · {meta['effort']} effort" if meta.get("effort") else ""
    if meta.get("operator") == "codex":
        name = meta.get("codex_label") or codex_models().get(meta.get("codex_model") or "", {}).get(
            "label") or meta.get("codex_model") or "?"
        provider = meta.get("codex_provider") or "hf"
        if provider == "openai":
            return f"Codex · {name} (OpenAI){effort}"
        if provider == "vendor":
            return f"Codex · {name} (own API){effort}"
        route = f", {meta['codex_route']}" if meta.get("codex_route") else ""
        return f"Codex · {name} (Hugging Face{route}){effort}"
    return f"Claude Code · {meta.get('claude_model') or 'default model'}{effort}"


# ------------------------------------------------------------------------------------- options
def allowed_tools(output_root: Path) -> list[str]:
    """The agentic batch's allow list (ab_common.sh allowed_tools), plus the UI's root, where this
    conversation's sessions are."""
    ui_root = "//" + str(Path(output_root).resolve()).lstrip("/")
    return [
        f"Bash(python3 -u {SKILL}/scripts/drive.py *)",
        f"Bash(python3 -u {SKILL}/scripts/inspect_session.py *)",
        "Bash(python3 experiments/agentic_batch/ab_helper.py *)",
        f"Read(./{SKILL}/**)", f"Glob(./{SKILL}/**)", f"Grep(./{SKILL}/**)",
        "Read(./experiments/agentic_batch/prompt_*.txt)",
        "Read(./output/fem_workflow_operator/**)", "Glob(./output/fem_workflow_operator/**)",
        "Grep(./output/fem_workflow_operator/**)",
        "Write(./output/fem_workflow_operator/*/operator/**)", "Edit(./output/fem_workflow_operator/*/operator/**)",
        "Write(./output/variations/**)", "Edit(./output/variations/**)",
        f"Read({ui_root}/**)", f"Glob({ui_root}/**)", f"Grep({ui_root}/**)",
        f"Write({ui_root}/*/operator/**)", f"Edit({ui_root}/*/operator/**)",
        "TodoWrite",
    ]


def disallowed_tools() -> list[str]:
    out = []
    for c in SHELL_READERS + ["python3 -", "python -"]:
        out += [f"Bash({c})", f"Bash({c} *)"]
    return out


def _operator_brief(meta: dict) -> str:
    """What every operator is told about the UI: the task, its guides, the user's model choices."""
    return (
        "You operate MechFoundry's FEM workflow for a user who is watching it in the Streamlit UI. "
        "They type in the chat on the left; the visualization on the right follows the session you are "
        "working on, live (geometry, regions, materials, solved fields, optimization iterations).\n\n"
        f"To simulate an object, follow {SKILL}/SKILL.md (read it first). "
        "The user sees the pictures in the UI, so report in a few lines (the verdict, the key "
        "numbers with units, every deviation) and give paths only when asked.\n\n"
        "The user chose the workflow's own models and compute environment in the UI: "
        f"text {meta.get('provider')}/{meta.get('model')}, vision {meta.get('vision_provider')}/"
        f"{meta.get('vision_model')}, compute environment {meta.get('compute_env')}. `drive.py start` "
        "uses these by default; do not pass other --provider, --model or --compute-env values unless "
        "the user asks."
    )


def system_note(meta: dict, upload_dir: Path) -> str:
    """Claude's: appended to the claude_code system prompt."""
    return _operator_brief(meta) + (
        "\n\n"
        "The user may send further prompts while you work, or stop you. After a stop, run "
        "`drive.py status SID` before anything else on a session that was mid-step. Anything outside "
        "the runner commands and the session files needs the user's approval, which the UI asks for; "
        "attempt such an action only when the user asked for it, and treat a denial as final. "
        f"Files the user attaches are saved in {upload_dir}."
    )


def codex_path_note(output_root: Path) -> str:
    """Codex's first developer note: where the repository is (ab_common.sh write_path_note says
    why a batch needs it; the UI's sessions are in its own root, which may be elsewhere)."""
    linked = ""
    if (REPO_ROOT / ".git").is_file() or "/.claude/worktrees/" in f"{REPO_ROOT}/":
        linked = (' It is a git worktree; the ".claude/worktrees/..." part is part of the repository '
                  "path, not a configuration folder.")
    return (f"Environment note: the repository root and working directory is {REPO_ROOT}.{linked} A "
            f"relative path resolves here. This UI's workflow sessions are in {output_root} (see the UI "
            "note).")


def codex_ui_note(meta: dict, upload_dir: Path, output_root: Path) -> str:
    """Codex's last developer note, after the tool note (codex_tools.txt): what the batch's own
    codex_note.txt says about the runner, and what differs in the UI (the user is there)."""
    return (
        "UI note: this note comes after the tool note and wins where they differ.\n\n"
        + _operator_brief(meta) + "\n\n"
        f"This UI keeps its workflow sessions in {output_root}: `drive.py start` creates them there "
        "(FEM_OPERATOR_OUTPUT_ROOT is set), so a session's folder is "
        f"{output_root}/<session_id>/ and you write under {output_root}/<session_id>/operator/, wherever "
        "the skill or the tool note says output/fem_workflow_operator/<session_id>/. drive.py prints "
        "the paths.\n\n"
        "drive.py and inspect_session.py: type each exactly as the skill writes it, `python3 -u "
        "src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py ...`, as one plain command from "
        "the repository root (no cd, &&, pipes, redirects or variables). Only that form runs with the GPU, "
        "the network and the session folders. A drive.py command can run for many minutes (an idle-GPU "
        "wait, generation, painting, a solve): when exec_command returns while it is still running, keep "
        "waiting on that same process with write_stdin (empty chars, a long yield_time_ms) until it exits, "
        "and read its whole output. Never start a second drive.py command for the session while one is "
        "running.\n\n"
        "Nobody answers during a reply, so rule 1 holds while you work; but the user reads your final "
        "message and may send further prompts after it, or stop you. After a stop, run "
        "`drive.py status SID` before anything else on a session that was mid-step. There are no "
        "approvals here: the runner's two command forms run with the GPU and the network, any other "
        "command runs in the sandbox (no network, writes only in the repository and the session "
        "folders) or is refused; do not try to work around a refusal. "
        f"Files the user attaches are saved in {upload_dir}; pictures among them come with the message."
    )


# --------------------------------------------------------------------------------- conversation
class _Conversation:
    """What every operator's conversation shares: its folder (meta.json, chat.jsonl, uploads/),
    the events the page draws, and the workflow sessions it touched. Every public method is safe
    to call from the Streamlit thread; the turn itself runs on the hub's loop."""

    operator = ""

    def __init__(self, hub: "AgentHub", directory: Path):
        self.hub = hub
        self.dir = Path(directory)
        self.id = self.dir.name
        self.events: list[dict] = self._read_chat()
        self.draft = {"text": "", "thinking": ""}
        self.busy = False
        self.pending: dict[str, asyncio.Future] = {}
        self.lock = threading.RLock()
        self.version = 0          # bumps on every change, for cheap polling

    # ---------------------------------------------------------------------------- storage
    @property
    def meta(self) -> dict:
        try:
            return json.loads((self.dir / "meta.json").read_text())
        except (OSError, ValueError):
            return {"id": self.id}

    def update_meta(self, **fields) -> dict:
        with self.lock:
            meta = self.meta
            meta.update(fields)
            meta["updated"] = time.time()
            tmp = self.dir / "meta.json.tmp"
            tmp.write_text(json.dumps(meta, indent=1))
            tmp.replace(self.dir / "meta.json")
            self.version += 1
            return meta

    def _read_chat(self) -> list[dict]:
        f = self.dir / "chat.jsonl"
        if not f.exists():
            return []
        out = []
        for line in f.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def _record(self, ev: dict, keep: bool = True) -> None:
        ev.setdefault("ts", time.time())
        with self.lock:
            if ev["type"] == "delta":
                key = "text" if ev.get("kind") == "text" else "thinking"
                self.draft[key] += ev.get("text", "")
            else:
                if ev["type"] in ("assistant_text", "tool_use", "thinking"):
                    self.draft = {"text": "", "thinking": ""}
                if keep:
                    self.events.append(ev)
                    with open(self.dir / "chat.jsonl", "a") as f:
                        f.write(json.dumps(ev, default=str) + "\n")
            self.version += 1

    def snapshot(self) -> dict:
        """What the page draws: a copy, so the loop can go on appending meanwhile."""
        with self.lock:
            return {"events": list(self.events), "draft": dict(self.draft), "busy": self.busy,
                    "pending": [e for e in self.events if e["type"] == "permission"
                                and e["id"] in self.pending],
                    "meta": self.meta, "version": self.version}

    # ------------------------------------------------------------------- from the page thread
    def send(self, text: str, attachments: list[Path] = ()) -> bool:
        """Queue a user message; False if a reply is still streaming."""
        with self.lock:
            if self.busy:
                return False
            self.busy = True
            self.version += 1
        self.hub.call(self._send(text, [Path(a) for a in attachments]))
        return True

    def interrupt(self) -> None:
        self.hub.call(self._interrupt())

    def answer_permission(self, request_id: str, allow: bool) -> None:
        def resolve():
            f = self.pending.get(request_id)
            if f is not None and not f.done():
                f.set_result(bool(allow))
        self.hub.loop.call_soon_threadsafe(resolve)

    def save_upload(self, name: str, data: bytes) -> Path:
        d = self.dir / "uploads"
        d.mkdir(exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._") or "file"
        target, k = d / safe, 1
        while target.exists():
            target = d / f"{Path(safe).stem}-{k}{Path(safe).suffix}"
            k += 1
        target.write_bytes(data)
        return target

    def _track(self, text: str) -> None:
        """The workflow session the agent is on: the last id a runner command named or printed."""
        found = SESSION_IN_COMMAND.findall(text) + SESSION_IN_RESULT.findall(text) + SESSION_LINE.findall(text)
        if not found:
            return
        sid = found[-1]
        if not (self.hub.output_root / sid).is_dir():
            return
        meta = self.meta
        sessions = list(meta.get("sessions", []))
        if sid not in sessions:
            sessions.append(sid)
        if sessions != meta.get("sessions") or meta.get("current_session") != sid:
            self.update_meta(sessions=sessions, current_session=sid)

    def operator_env(self, meta: dict) -> dict[str, str]:
        """The setup card's choices, which `drive.py start` takes as its defaults."""
        return {
            "FEM_OPERATOR_OUTPUT_ROOT": str(self.hub.output_root),
            "FEM_OPERATOR_PROVIDER": meta.get("provider") or "",
            "FEM_OPERATOR_MODEL": meta.get("model") or "",
            "FEM_OPERATOR_VISION_PROVIDER": meta.get("vision_provider") or "",
            "FEM_OPERATOR_VISION_MODEL": meta.get("vision_model") or "",
            "FEM_OPERATOR_COMPUTE_ENV": meta.get("compute_env") or "",
        }

    def _record_user(self, text: str, files: list[Path]) -> None:
        self._record({"type": "user", "text": text,
                      "attachments": [{"name": f.name, "path": str(f),
                                       "mime": mimetypes.guess_type(f.name)[0] or ""} for f in files]})

    async def close(self):
        pass


class AgentConversation(_Conversation):
    """One Claude conversation, through the Claude Agent SDK."""

    operator = "claude"

    def __init__(self, hub: "AgentHub", directory: Path):
        super().__init__(hub, directory)
        self.tool_inputs: dict[str, dict] = {}
        self.client = None
        self.task: asyncio.Task | None = None

    # ----------------------------------------------------------------------- on the loop
    def options(self):
        from claude_agent_sdk import ClaudeAgentOptions
        meta = self.meta
        env = {
            **self.operator_env(meta),
            # A runner step can wait for an idle GPU and then generate for minutes; the Bash
            # tool's 2-minute default would cut it off (same values as the agentic batch).
            "BASH_DEFAULT_TIMEOUT_MS": "1200000",
            "BASH_MAX_TIMEOUT_MS": "3600000",
        }
        kw: dict[str, Any] = dict(
            cwd=str(REPO_ROOT),
            add_dirs=[str(self.hub.output_root), str(self.dir / "uploads")],
            system_prompt={"type": "preset", "preset": "claude_code",
                           "append": system_note(meta, self.dir / "uploads")},
            tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep", "TodoWrite"],
            allowed_tools=allowed_tools(self.hub.output_root),
            disallowed_tools=disallowed_tools(),
            permission_mode="default",
            can_use_tool=self._can_use_tool,
            setting_sources=[],              # no user CLAUDE.md, hooks or settings leak in
            strict_mcp_config=True,
            include_partial_messages=True,
            # A Read of a rendered view returns the picture as base64 in one stream message;
            # the SDK's 1 MB default drops the conversation on a ~0.5 MB PNG.
            max_buffer_size=64 * 1024 * 1024,
            env=env,
            settings='{"autoMemoryEnabled": false}',
            model=meta.get("claude_model") or None,
            resume=meta.get("claude_session_id") or None,
        )
        if meta.get("effort"):
            kw["effort"] = meta["effort"]
        status = agent_status()
        if status.cli:
            kw["cli_path"] = status.cli
        return ClaudeAgentOptions(**kw)

    async def _connect(self):
        if self.client is None:
            factory = self.hub.client_factory
            if factory is None:
                from claude_agent_sdk import ClaudeSDKClient as factory
            self.client = factory(options=self.options())
            await self.client.connect()

    async def close(self):
        for f in self.pending.values():
            if not f.done():
                f.cancel()
        if self.client is not None:
            try:
                await self.client.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self.client = None

    async def _send(self, text: str, files: list[Path]):
        self._record_user(text, files)
        try:
            await self._connect()
            content = self._content(text, files)

            async def gen():
                yield {"type": "user", "message": {"role": "user", "content": content},
                       "parent_tool_use_id": None}
            await self.client.query(gen())
            async for msg in self.client.receive_response():
                self._handle(msg)
        except Exception as ex:  # noqa: BLE001
            self._record({"type": "error", "message": (
                f"{type(ex).__name__}: {ex}. The connection to Claude was reset; the conversation is "
                "kept. Send a message (for example \"continue\") to resume.")})
            await self.close()
        finally:
            with self.lock:
                self.busy = False
                self.draft = {"text": "", "thinking": ""}
                self.version += 1

    def _content(self, text: str, files: list[Path]) -> list[dict]:
        blocks: list[dict] = []
        for f in files:
            mime = mimetypes.guess_type(f.name)[0] or ""
            if mime in IMAGE_TYPES and f.stat().st_size < 5_000_000:
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime,
                                                           "data": base64.b64encode(f.read_bytes()).decode()}})
        body = text
        if files:
            body += "\n\n[Attached files: " + "; ".join(str(f) for f in files) + "]"
        blocks.append({"type": "text", "text": body})
        return blocks

    async def _interrupt(self):
        for f in self.pending.values():
            if not f.done():
                f.set_result(False)
        if self.client is not None and self.busy:
            try:
                await self.client.interrupt()
            except Exception as ex:  # noqa: BLE001
                self._record({"type": "error", "message": f"stop failed: {ex}"}, keep=False)

    def _handle(self, msg) -> None:
        from claude_agent_sdk import (AssistantMessage, ResultMessage, StreamEvent, SystemMessage,
                                      TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock, UserMessage)
        if isinstance(msg, StreamEvent):
            if msg.parent_tool_use_id:
                return
            d = (msg.event or {}).get("delta") or {}
            if (msg.event or {}).get("type") == "content_block_delta":
                if d.get("type") == "text_delta":
                    self._record({"type": "delta", "kind": "text", "text": d.get("text", "")}, keep=False)
                elif d.get("type") == "thinking_delta":
                    self._record({"type": "delta", "kind": "thinking", "text": d.get("thinking", "")}, keep=False)
        elif isinstance(msg, SystemMessage):
            sid = (msg.data or {}).get("session_id")
            if msg.subtype == "init" and sid:
                self.update_meta(claude_session_id=sid)
        elif isinstance(msg, AssistantMessage):
            for b in msg.content:
                if isinstance(b, TextBlock) and b.text.strip():
                    self._record({"type": "assistant_text", "text": b.text, "sub": msg.parent_tool_use_id})
                elif isinstance(b, ThinkingBlock) and b.thinking.strip():
                    self._record({"type": "thinking", "text": b.thinking, "sub": msg.parent_tool_use_id})
                elif isinstance(b, ToolUseBlock):
                    self.tool_inputs[b.id] = {"name": b.name, "input": b.input}
                    self._record({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input,
                                  "sub": msg.parent_tool_use_id})
                    self._track(str((b.input or {}).get("command", "")) if b.name == "Bash" else "")
            if msg.error:
                self._record({"type": "error", "message": f"API error: {msg.error}"})
        elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
            for b in msg.content:
                if isinstance(b, ToolResultBlock):
                    self._record(self._tool_result(b))
        elif isinstance(msg, ResultMessage):
            if msg.session_id:
                self.update_meta(claude_session_id=msg.session_id)
            self._record({"type": "result", "is_error": msg.is_error, "duration_ms": msg.duration_ms,
                          "cost_usd": msg.total_cost_usd, "reason": getattr(msg, "terminal_reason", None)})

    def _tool_result(self, b) -> dict:
        text = ""
        if isinstance(b.content, str):
            text = b.content
        elif isinstance(b.content, list):
            text = "".join(c.get("text", "") for c in b.content if c.get("type") == "text")
        use = self.tool_inputs.get(b.tool_use_id, {})
        ev = {"type": "tool_result", "tool_use_id": b.tool_use_id, "is_error": bool(b.is_error),
              "text": text[:20000]}
        if use.get("name") == "Bash":
            self._track(text)
        if use.get("name") == "Read":
            fp = str((use.get("input") or {}).get("file_path", ""))
            if fp.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
                p = Path(fp) if Path(fp).is_absolute() else REPO_ROOT / fp
                ev["image"] = str(p)
        return ev

    async def _can_use_tool(self, name: str, tool_input: dict, context):
        from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny
        rid = secrets.token_hex(6)
        fut = asyncio.get_running_loop().create_future()
        self.pending[rid] = fut
        self._record({"type": "permission", "id": rid, "tool": name, "input": tool_input})
        try:
            ok = await asyncio.wait_for(fut, timeout=1800)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            ok = False
        finally:
            self.pending.pop(rid, None)
        self._record({"type": "permission_done", "id": rid, "allowed": bool(ok)})
        if ok:
            return PermissionResultAllow()
        return PermissionResultDeny(message="The user denied this action in the UI.")


def _die_with_parent() -> None:
    """preexec_fn: the child gets SIGTERM when the thread that started it ends (Linux prctl
    PR_SET_PDEATHSIG). Children are started on the hub's loop thread, which lives as long as the
    server, so a killed or crashed server leaves no relay or Codex process behind."""
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
        except Exception:  # noqa: BLE001
            pass


def _shell_command(command: str) -> str:
    """The command the agent typed: Codex reports it wrapped as ``/bin/bash -lc '<command>'``."""
    return OPS.unwrap(command)


class CodexConversation(_Conversation):
    """One Codex conversation: each user message is one ``codex exec --json`` process (the first
    starts the thread, each later one is ``codex exec resume <thread>``), its event stream turned
    into the events the page draws.

    Codex runs as the agentic batch runs it (ab_common.sh ab_launch_codex): the conversation's own
    CODEX_HOME written by ``ab_helper.py codex-home`` (provider through the hub's hf_relay.py,
    approval never, the workspace-write sandbox, the rules that let only ``drive.py`` and
    ``inspect_session.py`` out of it), a closed stdin, and no inherited CLAUDE*/CODEX* variable.
    Nothing asks the user, so there are no permission events; Stop terminates the process."""

    operator = "codex"

    def __init__(self, hub: "AgentHub", directory: Path):
        super().__init__(hub, directory)
        self.proc: asyncio.subprocess.Process | None = None
        self.stopping = False
        self.rollout: Path | None = None
        self.rollout_pos = 0
        # Codex numbers a process's items item_0, item_1, ... afresh every turn; the page pairs a
        # tool_use with its result by id, so ids are made unique with a per-turn prefix.
        self.turn = ""
        self.started: set[str] = set()

    @property
    def home(self) -> Path:
        return self.dir / "codex_home"

    # ----------------------------------------------------------------------- the launch
    def notes(self) -> list[Path]:
        """The developer notes, in order: where the repository is, the batch's tool-name note,
        and the UI's own (which wins where they differ). Rewritten every turn, as the setup can
        only be read from meta.json."""
        d = self.dir / "notes"
        d.mkdir(exist_ok=True)
        path_note, ui_note = d / "path_note.txt", d / "ui_note.txt"
        path_note.write_text(codex_path_note(self.hub.output_root) + "\n")
        ui_note.write_text(codex_ui_note(self.meta, self.dir / "uploads", self.hub.output_root) + "\n")
        return [path_note, CODEX_TOOLS, ui_note]

    def home_command(self, base_url: str) -> tuple[list[str], dict[str, str]]:
        """``ab_helper.py codex-home`` for this conversation, and its environment: the relay's
        URL (AB_CODEX_BASE_URL) is what points config.toml at it."""
        meta = self.meta
        cmd = [sys.executable, str(AB_HELPER), "codex-home", "--dir", str(self.home),
               "--model", meta.get("codex_model") or "", "--provider", meta.get("codex_provider") or "hf"]
        if meta.get("effort"):
            cmd += ["--effort", meta["effort"]]
        for note in self.notes():
            cmd += ["--note", str(note)]
        env = {k: v for k, v in os.environ.items() if k != "AB_CODEX_BASE_URL"}
        if base_url:
            env["AB_CODEX_BASE_URL"] = base_url
        return cmd, env

    async def write_home(self, base_url: str) -> dict:
        """config.toml and rules/ for this turn (the relay's port can change between turns, after
        a server restart), and the model they resolve to, recorded in meta.json. An OpenAI model
        on the user's Codex login gets the same configuration from operators.write_openai_home."""
        if self.provider == "openai":
            model = OPS.write_openai_home(self.home, self.meta.get("codex_model") or "",
                                          self.meta.get("effort") or None, self.notes())
            self.update_meta(codex_model_id=model["model"], codex_label=model["label"], codex_route=None,
                             codex_provider="openai", codex_price=None)
            return model
        cmd, env = self.home_command(base_url)
        proc = await asyncio.create_subprocess_exec(*cmd, env=env, stdin=subprocess.DEVNULL,
                                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ab_helper.py codex-home failed: {err.decode(errors='replace')[-500:]}")
        model = json.loads(out.decode().strip().splitlines()[-1])
        self.update_meta(codex_model_id=model.get("model"), codex_label=model.get("label"),
                         codex_route=model.get("route"), codex_service=model.get("provider"),
                         codex_price=model.get("price"))
        return model

    def command(self, text: str, files: list[Path]) -> list[str]:
        """The turn's command line. Pictures go as ``-i`` images; every attachment is also named
        in the text, as for Claude. ``--`` ends the options: ``exec``'s ``-i`` takes any number of
        files and would swallow the prompt (checked on codex-cli 0.163.0, both forms)."""
        meta = self.meta
        cli = codex_status().cli or os.environ.get("CODEX_CLI") or "codex"
        thread = meta.get("codex_thread_id")
        cmd = [cli, "exec"] + (["resume"] if thread else []) + ["--json"]
        if not thread:
            cmd += ["-C", str(REPO_ROOT)]      # a resumed thread keeps its working root
        # The UI's root may lie outside the repository, and the agent writes its brief and plan
        # there with apply_patch, which runs in the sandbox. `resume` takes no --add-dir; -c does.
        roots = sorted({str(self.hub.output_root), str(self.dir / "uploads")})
        cmd += ["-c", f"sandbox_workspace_write.writable_roots={json.dumps(roots)}"]
        for f in files:
            if (mimetypes.guess_type(f.name)[0] or "") in IMAGE_TYPES:
                cmd += ["-i", str(f)]
        body = text
        if files:
            body += "\n\n[Attached files: " + "; ".join(str(f) for f in files) + "]"
        return cmd + ["--"] + ([thread] if thread else []) + [body]

    @property
    def provider(self) -> str:
        """hf (Hugging Face, through the relay), vendor (the company's own API) or openai (the
        user's Codex login)."""
        return self.meta.get("codex_provider") or "hf"

    def provider_key(self) -> tuple[str, str]:
        """(key variable, its value) the provider needs; ("", "") for the Codex login."""
        if self.provider == "openai":
            return "", ""
        if self.provider == "hf":
            return "HF_TOKEN", hf_token()[0]
        try:
            var = ab_helper().codex_model(self.meta.get("codex_model") or "", "vendor")["env_key"]
        except (SystemExit, Exception):  # noqa: BLE001
            return "", ""
        return var, OPS.provider_key(var)[0]

    def env(self, token: str) -> dict[str, str]:
        """The CLI's environment: this one without CLAUDE*/CODEX* (a parent's Codex or Claude
        session must not leak in; ab_common.sh clean_env_args), plus its CODEX_HOME, the setup
        card's choices and the Hugging Face token (to the provider only: Codex keeps *TOKEN*
        variables out of the agent's commands)."""
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("CLAUDE", "CODEX")) and k != "AB_CODEX_BASE_URL"}
        env.update(self.operator_env(self.meta))
        env["CODEX_HOME"] = str(self.home)
        var = self.provider_key()[0]
        if var:
            env[var] = token
        return env

    # ----------------------------------------------------------------------- on the loop
    async def _send(self, text: str, files: list[Path]):
        self._record_user(text, files)
        t0, outcome = time.time(), {}
        ticker = None
        try:
            var, token = self.provider_key()
            if var and not token:
                raise RuntimeError("no Hugging Face token (set HF_TOKEN or run `hf auth login`)" if var == "HF_TOKEN"
                                   else f"{var} is not set (export it, or set it in mechfoundry-agents' conda variables)")
            await self.write_home(await self.hub.hf_relay() if self.provider == "hf" else "")
            self.stopping = False
            self.turn, self.started = secrets.token_hex(3), set()
            self._rollout_mark()
            self.home.mkdir(exist_ok=True)
            with open(self.dir / "codex_stderr.log", "ab") as err:
                self.proc = await asyncio.create_subprocess_exec(
                    *self.command(text, files), cwd=str(REPO_ROOT), env=self.env(token),
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err,
                    limit=CODEX_LINE_LIMIT, preexec_fn=_die_with_parent)
            ticker = asyncio.ensure_future(self._tick())
            async for line in self.proc.stdout:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                self._handle(rec, outcome)
                self._poll_rollout()
            rc = await self.proc.wait()
            self._poll_rollout()
            if self.provider == "openai":
                OPS.sync_login(self.home, into_home=False)
            self._finish(rc, outcome, time.time() - t0)
        except Exception as ex:  # noqa: BLE001
            self._record({"type": "error", "message": (
                f"{type(ex).__name__}: {ex}. The conversation is kept; send a message to try again.")})
        finally:
            if ticker is not None:
                ticker.cancel()
            if self.proc is not None and self.proc.returncode is None:
                self.proc.kill()
            self.proc = None
            with self.lock:
                self.busy = False
                self.draft = {"text": "", "thinking": ""}
                self.version += 1

    def _handle(self, rec: dict, outcome: dict) -> None:
        """One ``codex exec --json`` event (shapes as measured on codex-cli 0.163.0)."""
        kind = rec.get("type")
        item = rec.get("item") if isinstance(rec.get("item"), dict) else {}
        itype = item.get("type")
        iid = f"{self.turn}-{item.get('id')}"
        if kind == "thread.started" and rec.get("thread_id"):
            if self.meta.get("codex_thread_id") != rec["thread_id"]:
                self.update_meta(codex_thread_id=rec["thread_id"])
        elif kind in ("item.started", "item.completed") and itype == "command_execution":
            command = _shell_command(str(item.get("command") or ""))
            if iid not in self.started:
                self.started.add(iid)
                # A file read shows as Read, ls as Glob, grep as Grep, as Claude's would.
                name, tool_input = OPS.classify(command)
                self._record({"type": "tool_use", "id": iid, "name": name, "input": tool_input})
            if kind == "item.started":
                return
            output = str(item.get("aggregated_output") or "")
            code = item.get("exit_code")
            self._record({"type": "tool_result", "tool_use_id": iid,
                          "is_error": code not in (0, None) or item.get("status") == "failed",
                          "text": output[:20000] + ("" if code in (0, None) else f"\n[exit code {code}]")})
            self._track(command)
            self._track(output)
        elif kind == "item.completed" and itype == "file_change":
            changes = item.get("changes") or []
            # Write when it only adds files, else Edit, as Claude's would show (operators.py).
            name, tool_input = OPS.file_change_tool(changes)
            self._record({"type": "tool_use", "id": iid, "name": name, "input": tool_input})
            self._record({"type": "tool_result", "tool_use_id": iid,
                          "is_error": item.get("status") == "failed",
                          "text": "\n".join(f"{c.get('kind', '')} {c.get('path', '')}" for c in changes)})
        elif kind == "item.completed" and itype == "agent_message" and str(item.get("text") or "").strip():
            self._record({"type": "assistant_text", "text": item["text"]})
        elif kind == "item.completed" and itype == "reasoning" and str(item.get("text") or "").strip():
            self._record({"type": "thinking", "text": item["text"]})
        elif kind == "item.completed" and itype == "error":
            message = str(item.get("message") or "")
            quiet = message.startswith(CODEX_QUIET_ERRORS)
            self._record({"type": "notice" if quiet else "error", "message": message})
        elif kind == "error":
            # "Reconnecting... 1/5": Codex retries by itself; kept, not drawn (turn.failed is).
            self._record({"type": "notice", "message": str(rec.get("message") or "")})
        elif kind == "turn.completed":
            outcome["usage"] = rec.get("usage") or {}
        elif kind == "turn.failed":
            outcome["failed"] = OPS.short_error((rec.get("error") or {}).get("message") or "turn failed")
            self._record({"type": "error", "message": f"Codex: {outcome['failed']}"})

    def _finish(self, rc: int, outcome: dict, seconds: float) -> None:
        """The turn's closing line: its time and priced tokens (ab_helper.codex_cost, at the
        model's codex_models.json price), or why it ended."""
        usage = outcome.get("usage")
        cost = None
        if usage:
            cached, written = usage.get("cached_input_tokens", 0), usage.get("cache_write_input_tokens", 0)
            tokens = {"input_tokens": max(0, usage.get("input_tokens", 0) - cached - written),
                      "cache_read_input_tokens": cached, "cache_creation_input_tokens": written,
                      "output_tokens": usage.get("output_tokens", 0)}
            try:
                cost = ab_helper().codex_cost(tokens, self.meta.get("codex_price"))
            except Exception:  # noqa: BLE001
                cost = None
        reason = "aborted_by_user" if self.stopping else None
        is_error = not self.stopping and bool(outcome.get("failed") or rc != 0 or not usage)
        if is_error and not outcome.get("failed"):
            try:
                tail = (self.dir / "codex_stderr.log").read_text(errors="replace")[-800:].strip()
            except OSError:
                tail = ""
            self._record({"type": "error", "message": f"Codex exited with code {rc} before finishing "
                                                      f"its reply. {tail}".strip()})
        self._record({"type": "result", "is_error": is_error, "duration_ms": int(seconds * 1000),
                      "cost_usd": cost, "reason": reason, "usage": usage, "exit_code": rc})

    # ------------------------------------------------------------------ pictures it looked at
    def _rollout_mark(self) -> None:
        """Remember where the thread's rollout ends, so this turn shows only its own pictures."""
        self.rollout = self._find_rollout()
        self.rollout_pos = self.rollout.stat().st_size if self.rollout else 0

    def _find_rollout(self) -> Path | None:
        thread = self.meta.get("codex_thread_id")
        if not thread:
            return None
        hits = sorted(self.home.glob(f"sessions/**/rollout-*{thread}.jsonl"))
        return hits[-1] if hits else None

    def _poll_rollout(self) -> None:
        """The stream does not report view_image calls; the rollout under CODEX_HOME/sessions
        does. Each new one becomes a tool_use and a result carrying the picture, as a Claude
        Read of a picture does."""
        if self.rollout is None:
            self.rollout, self.rollout_pos = self._find_rollout(), 0
            if self.rollout is None:
                return
        try:
            with open(self.rollout, "rb") as fh:
                fh.seek(self.rollout_pos)
                chunk = fh.read()
        except OSError:
            return
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        self.rollout_pos += end + 1
        for line in chunk[:end].splitlines():
            if b'"view_image"' not in line:
                continue
            try:
                p = json.loads(line).get("payload") or {}
                if p.get("type") != "function_call" or p.get("name") != "view_image":
                    continue
                path = str(json.loads(p.get("arguments") or "{}").get("path", ""))
            except (ValueError, AttributeError):
                continue
            call = p.get("call_id") or secrets.token_hex(4)
            full = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
            self._record({"type": "tool_use", "id": call, "name": "view_image", "input": {"file_path": path}})
            self._record({"type": "tool_result", "tool_use_id": call, "is_error": not full.exists(),
                          "text": "", "image": str(full)})

    async def _tick(self):
        """While a command runs nothing comes on the stream; the rollout is read meanwhile."""
        while True:
            await asyncio.sleep(1.0)
            self._poll_rollout()

    # ----------------------------------------------------------------------------- stop
    async def _interrupt(self):
        """Stop the reply: terminate the Codex process (a drive.py it started keeps running, as
        under Claude; a running solve is stopped with Stop FEA). The thread is kept, and the next
        message resumes it."""
        proc = self.proc
        if proc is None or proc.returncode is not None:
            return
        self.stopping = True
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), 10)
        except asyncio.TimeoutError:
            proc.kill()

    async def close(self):
        await self._interrupt()


# ------------------------------------------------------------------------------------------ hub
class AgentHub:
    """Every agent conversation of this server process, and the loop they run on."""

    def __init__(self, output_root: Path, client_factory=None):
        self.output_root = Path(output_root).resolve()
        self.root = self.output_root.parent / f"{self.output_root.name}_agent"
        self.client_factory = client_factory
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="agent-sessions", daemon=True)
        self.thread.start()
        self.conversations: dict[str, _Conversation] = {}
        self._lock = threading.Lock()
        # One hf_relay.py for every Codex conversation of this server, started at the first
        # Codex message (on the loop thread, see _die_with_parent) and stopped with the hub.
        self.relay: subprocess.Popen | None = None
        self.relay_url = ""
        self._relay_lock: asyncio.Lock | None = None
        atexit.register(self.shutdown)

    def call(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def create(self, *, provider: str, model: str, vision_provider: str, vision_model: str,
               compute_env: str, claude_model: str = "", effort: str = "", operator: str = "claude",
               codex_model: str = "", codex_provider: str = "hf") -> _Conversation:
        """A new conversation. ``provider``/``model``/``vision_*``/``compute_env`` are the
        workflow's own choices (``drive.py start``'s defaults); the operator runs Claude
        ``claude_model`` or, for ``operator="codex"``, the codex_models.json entry
        ``codex_model`` through ``codex_provider`` (hf, vendor, or openai with ``codex_model`` an OpenAI
        model of the user's Codex login; operators.py); ``effort`` is the operator's reasoning effort."""
        if operator not in OPERATORS:
            raise ValueError(f"unknown operator {operator!r}")
        cid = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        d = self.root / cid
        (d / "uploads").mkdir(parents=True)
        meta = {"id": cid, "title": "Agent session", "created": time.time(), "updated": time.time(),
                "operator": operator,
                "provider": provider, "model": model, "vision_provider": vision_provider,
                "vision_model": vision_model, "compute_env": compute_env, "claude_model": claude_model,
                "effort": effort, "claude_session_id": None, "sessions": [], "current_session": None}
        if operator == "codex":
            if codex_provider == "openai":
                entry = next((m for m in OPS.openai_models() if m["slug"] == codex_model), {})
                meta.update(claude_model="", codex_model=codex_model, codex_label=entry.get("label"),
                            codex_route=None, codex_provider="openai", codex_thread_id=None)
            else:
                entry = codex_models().get(codex_model) or {}
                meta.update(claude_model="", codex_model=codex_model, codex_label=entry.get("label"),
                            codex_route=entry.get("route") if codex_provider == "hf" else None,
                            codex_provider=codex_provider, codex_thread_id=None)
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        return self.get(cid)

    def get(self, cid: str) -> _Conversation:
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{4}", cid or ""):
            raise KeyError(cid)
        with self._lock:
            if cid not in self.conversations:
                d = self.root / cid
                if not (d / "meta.json").exists():
                    raise KeyError(cid)
                try:
                    operator = json.loads((d / "meta.json").read_text()).get("operator")
                except (OSError, ValueError):
                    operator = None
                # Conversations from before the operator choice are Claude's.
                cls = CodexConversation if operator == "codex" else AgentConversation
                self.conversations[cid] = cls(self, d)
            return self.conversations[cid]

    def list(self) -> list[dict]:
        out = []
        if self.root.is_dir():
            for d in self.root.iterdir():
                try:
                    out.append(json.loads((d / "meta.json").read_text()))
                except (OSError, ValueError):
                    continue
        return sorted(out, key=lambda m: m.get("updated", 0), reverse=True)

    def owned_sessions(self) -> dict[str, str]:
        """Workflow session id -> the agent conversation that drives it."""
        return {sid: m["id"] for m in self.list() for sid in m.get("sessions", [])}

    def delete(self, cid: str) -> None:
        conv = self.get(cid)
        if conv.busy:
            raise RuntimeError("stop the agent first")
        self.call(conv.close()).result(timeout=10)
        self.conversations.pop(cid, None)
        shutil.rmtree(conv.dir)

    # ------------------------------------------------------------------------- the HF relay
    async def hf_relay(self) -> str:
        """The relay's base URL, starting it on first use (ab_common.sh start_hf_relay): Codex
        speaks only the Responses API, and Hugging Face's own Responses endpoint refuses Codex's
        requests (hf_relay.py's docstring), so Codex is pointed at this local translator."""
        if self._relay_lock is None:
            self._relay_lock = asyncio.Lock()
        async with self._relay_lock:
            if self.relay is not None and self.relay.poll() is None:
                return self.relay_url
            d = self.root / ".hf_relay"
            d.mkdir(parents=True, exist_ok=True)
            port_file = d / "port"
            port_file.unlink(missing_ok=True)
            with open(d / "relay.err", "ab") as err:
                # An interpreter with Pillow, which the relay needs to shrink pictures
                # (operators.relay_python).
                self.relay = subprocess.Popen(
                    [OPS.relay_python(), str(HF_RELAY), "--upstream", HF_UPSTREAM,
                     "--port-file", str(port_file), "--log", str(d / "relay.log")],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err,
                    preexec_fn=_die_with_parent)
            for _ in range(100):
                if port_file.exists() and port_file.read_text().strip():
                    break
                if self.relay.poll() is not None:
                    break
                await asyncio.sleep(0.1)
            port = port_file.read_text().strip() if port_file.exists() else ""
            if not port:
                self._stop_relay()
                raise RuntimeError(f"hf_relay.py did not start (see {d / 'relay.err'})")
            self.relay_url = f"http://127.0.0.1:{port}/v1"
            return self.relay_url

    def _stop_relay(self) -> None:
        relay, self.relay, self.relay_url = self.relay, None, ""
        if relay is not None and relay.poll() is None:
            relay.terminate()
            try:
                relay.wait(5)
            except subprocess.TimeoutExpired:
                relay.kill()

    def shutdown(self) -> None:
        """Stop every running Codex reply and the relay (at exit, or by a test)."""
        for conv in list(self.conversations.values()):
            proc = getattr(conv, "proc", None)
            if proc is not None and proc.returncode is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
        self._stop_relay()


def session_signature(session_dir: Path) -> tuple:
    """Changes whenever the agent's step changed something the page draws."""
    names = ("session.json", "operator/last_run.json", "operator/last_event.json",
             "operator/current_step.json", "boundary_conditions/resolved.glb",
             "materials/parts/materials.json", "solve/displacement_magnitude.glb")
    stamps = []
    for n in names:
        try:
            stamps.append((Path(session_dir) / n).stat().st_mtime_ns)
        except OSError:
            stamps.append(0)
    return tuple(stamps)
