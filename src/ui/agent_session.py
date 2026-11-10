"""Agent-controlled sessions: a Claude Code conversation that drives the workflow for the user.

The UI's setup card picks the workflow's text and vision models and the compute environment, as
for any conversation. "Start agent session" then hands the work to Claude Code (the locally
installed ``claude`` CLI, on the user's own login, through the Claude Agent SDK), which follows the
operator skill (``src/agents_wrapper/fem_workflow/operator_skill/SKILL.md``) and drives
``drive.py``. The page shows the session the agent is working on, read-only, and refreshes as its
files change; the user can send further prompts, stop a reply, and approve or deny any action
outside the allow list.

Streamlit-free, so it can be tested: one background thread runs an asyncio loop that owns every
SDK client; the page talks to it through thread-safe calls and reads plain snapshots.

Storage: ``<output root>_agent/<conversation id>/``
  meta.json   providers, models, compute env, Claude session id, the workflow sessions touched
  chat.jsonl  the conversation as events (text deltas are not kept)

The agent's workflow sessions live in the UI's own output root (``FEM_OPERATOR_OUTPUT_ROOT``), so
they are listed in the sidebar like any other; ``drive.py start`` takes the setup card's choices
from ``FEM_OPERATOR_PROVIDER`` and its siblings.
"""
from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL = "src/agents_wrapper/fem_workflow/operator_skill"
IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
CLAUDE_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
# The shell's file readers and stdin Python, refused outright as in the agentic batch
# (experiments/agentic_batch/ab_common.sh, denied_tools): the skill reads with the Read tool.
SHELL_READERS = ("cat head tail less more grep egrep fgrep rg find sed awk ls tree file stat wc od xxd "
                 "strings sort uniq cut diff cmp nl tac realpath readlink du git rm sudo conda pip").split()
SESSION_IN_COMMAND = re.compile(r"(?:drive|inspect_session)\.py\s+(?:--\S+\s+\S+\s+)*\w[\w-]*\s+([0-9a-f]{12})\b")
SESSION_IN_RESULT = re.compile(r'"session_id"\s*:\s*"([0-9a-f]{12})"')
SESSION_LINE = re.compile(r"session_id: ([0-9a-f]{12})")


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


def system_note(meta: dict, upload_dir: Path) -> str:
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
        "the user asks.\n\n"
        "The user may send further prompts while you work, or stop you. After a stop, run "
        "`drive.py status SID` before anything else on a session that was mid-step. Anything outside "
        "the runner commands and the session files needs the user's approval, which the UI asks for; "
        "attempt such an action only when the user asked for it, and treat a denial as final. "
        f"Files the user attaches are saved in {upload_dir}."
    )


# --------------------------------------------------------------------------------- conversation
class AgentConversation:
    """One Claude conversation. Every method is safe to call from the Streamlit thread."""

    def __init__(self, hub: "AgentHub", directory: Path):
        self.hub = hub
        self.dir = Path(directory)
        self.id = self.dir.name
        self.events: list[dict] = self._read_chat()
        self.draft = {"text": "", "thinking": ""}
        self.busy = False
        self.pending: dict[str, asyncio.Future] = {}
        self.tool_inputs: dict[str, dict] = {}
        self.client = None
        self.task: asyncio.Task | None = None
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

    # ----------------------------------------------------------------------- on the loop
    def options(self):
        from claude_agent_sdk import ClaudeAgentOptions
        meta = self.meta
        env = {
            "FEM_OPERATOR_OUTPUT_ROOT": str(self.hub.output_root),
            "FEM_OPERATOR_PROVIDER": meta.get("provider") or "",
            "FEM_OPERATOR_MODEL": meta.get("model") or "",
            "FEM_OPERATOR_VISION_PROVIDER": meta.get("vision_provider") or "",
            "FEM_OPERATOR_VISION_MODEL": meta.get("vision_model") or "",
            "FEM_OPERATOR_COMPUTE_ENV": meta.get("compute_env") or "",
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
        self._record({"type": "user", "text": text,
                      "attachments": [{"name": f.name, "path": str(f),
                                       "mime": mimetypes.guess_type(f.name)[0] or ""} for f in files]})
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
        self.conversations: dict[str, AgentConversation] = {}
        self._lock = threading.Lock()

    def call(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def create(self, *, provider: str, model: str, vision_provider: str, vision_model: str,
               compute_env: str, claude_model: str = "", effort: str = "") -> AgentConversation:
        cid = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        d = self.root / cid
        (d / "uploads").mkdir(parents=True)
        meta = {"id": cid, "title": "Agent session", "created": time.time(), "updated": time.time(),
                "provider": provider, "model": model, "vision_provider": vision_provider,
                "vision_model": vision_model, "compute_env": compute_env, "claude_model": claude_model,
                "effort": effort, "claude_session_id": None, "sessions": [], "current_session": None}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        return self.get(cid)

    def get(self, cid: str) -> AgentConversation:
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{4}", cid or ""):
            raise KeyError(cid)
        with self._lock:
            if cid not in self.conversations:
                d = self.root / cid
                if not (d / "meta.json").exists():
                    raise KeyError(cid)
                self.conversations[cid] = AgentConversation(self, d)
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
