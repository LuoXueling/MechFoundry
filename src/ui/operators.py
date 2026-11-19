"""The AI operators an agent session can run, and the small pieces every front end needs to run
them: which models exist here, where their keys are, and how to read what a Codex agent did.

Standard library only and free of Streamlit, so the UI (``agent_session.py``) and other front ends
on this checkout (MechFoundry Print Studio loads this file by path) share one implementation.

Operators. An operator id is a Claude model id (Claude Code, through the Claude Agent SDK; the
current models of the user's login, as Claude Code lists them when the SDK connects), or
``codex:<model>@<provider>`` for the Codex CLI:
  - provider ``hf``: an open-weight model of ``experiments/agentic_batch/codex_models.json`` through
    Hugging Face Inference Providers, pinned to its route (through ``hf_relay.py``);
  - provider ``vendor``: the same model on its company's own API (DeepSeek, Moonshot);
  - provider ``openai``: an OpenAI model on the user's own Codex login (``~/.codex/auth.json``),
    listed from Codex's own catalog for that login (``codex debug models``).
``operators()`` lists them all, each with why it cannot run here when it cannot.

Codex output. ``codex exec --json`` reports each shell command wrapped as ``/bin/bash -lc '<cmd>'``;
``unwrap`` gives the command typed, ``classify`` names it as Claude's tool would be (a file read as
Read, ``ls`` as Glob, ``grep`` as Grep), and ``short_error`` gives a provider's error in a line.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AB_DIR = REPO_ROOT / "experiments" / "agentic_batch"
AB_HELPER = AB_DIR / "ab_helper.py"
HF_RELAY = AB_DIR / "hf_relay.py"
CODEX_TOOLS = AB_DIR / "codex_tools.txt"   # the skill's Claude Code tool names -> Codex's (any Codex use)
CODEX_NOTE = AB_DIR / "codex_note.txt"     # the unattended batch's own rules (run_batch.sh only)
CODEX_MODELS_FILE = AB_DIR / "codex_models.json"
HF_UPSTREAM = "https://router.huggingface.co/v1"
AGENTS_ENV = "mechfoundry-agents"     # where keys the batch uses are kept (conda env config vars)

# The fallback when Claude Code cannot be asked (claude_models): the current models as of 2026-11.
CLAUDE_MODELS = (("claude-opus-5-5", "Claude Opus 5.5"), ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
                 ("claude-haiku-5-5", "Claude Haiku 5.5"), ("claude-fable-5-2", "Claude Fable 5.2"))
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
# What the open-weight models accept (codex_models.json's _comment); Codex sends it as
# model_reasoning_effort.
OPEN_EFFORTS = ("low", "medium", "high")
# "ultra" is "automatic task delegation": Codex's multi-agent tools, which every run turns off.
OPENAI_EFFORT_SKIP = ("ultra",)
GROUPS = {"claude": "Anthropic (Claude Code)", "open": "Open-weight (Codex)", "openai": "OpenAI (Codex)"}
# The company APIs codex_models.json's `vendor` names (ab_helper.CODEX_PROVIDERS).
VENDORS = {"deepseek": ("DeepSeek API", "DEEPSEEK_API_KEY"), "moonshot": ("Moonshot API", "MOONSHOT_API_KEY")}


# ---------------------------------------------------------------------------------- tables
_AB_HELPER_MODULE = None


def ab_helper():
    """experiments/agentic_batch/ab_helper.py as a module (standard library only), loaded by path,
    since experiments/ is not a package."""
    global _AB_HELPER_MODULE
    if _AB_HELPER_MODULE is None:
        spec = importlib.util.spec_from_file_location("ab_helper", AB_HELPER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _AB_HELPER_MODULE = module
    return _AB_HELPER_MODULE


def codex_models() -> dict[str, dict]:
    """codex_models.json's models: key -> {label, effort, price, ...}; {} when the file is absent."""
    try:
        return dict(ab_helper().codex_models())
    except (OSError, ValueError, KeyError, AttributeError):
        return {}


def claude_cli() -> str | None:
    cli = os.environ.get("CLAUDE_CLI") or shutil.which("claude")
    return cli if cli and Path(cli).exists() else None


def codex_cli() -> str | None:
    cli = os.environ.get("CODEX_CLI") or shutil.which("codex")
    return cli if cli and Path(cli).exists() else None


# ------------------------------------------------------------------------------------ keys
def hf_token() -> tuple[str, str]:
    """The Hugging Face token and where it came from: ``HF_TOKEN``, else the token file
    ``hf auth login`` writes (``HF_TOKEN_PATH``, ``$HF_HOME/token`` or
    ``~/.cache/huggingface/token``), as ab_common.sh codex_setup reads it. ("", "") if none."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip(), "HF_TOKEN"
    f = Path(os.environ.get("HF_TOKEN_PATH") or
             Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "token").expanduser()
    try:
        value = f.read_text().strip()
    except OSError:
        return "", ""
    return (value, str(f)) if value else ("", "")


def provider_key(env_key: str) -> tuple[str, str]:
    """(the key, where it came from) for a provider's key variable, as run_batch.sh finds it: the
    environment, then HF_TOKEN's token file, any other from the mechfoundry-agents conda
    environment's variables. ("", "") when there is none."""
    if env_key == "HF_TOKEN":
        return hf_token()
    if os.environ.get(env_key):
        return os.environ[env_key], "the environment"
    try:
        value = ab_helper().conda_env_var(AGENTS_ENV, env_key)
    except Exception:  # noqa: BLE001
        value = ""
    return (value, f"{AGENTS_ENV}'s conda variables") if value else ("", "")


def codex_login_file() -> Path:
    """The user's own Codex login (a ChatGPT sign-in or an API key); CODEX_LOGIN_HOME overrides
    ~/.codex."""
    return Path(os.environ.get("CODEX_LOGIN_HOME") or Path.home() / ".codex") / "auth.json"


def sync_login(home: Path, into_home: bool) -> None:
    """Share the user's Codex login with a private CODEX_HOME. Before a turn it is copied in; after
    the turn, a copy whose tokens Codex refreshed meanwhile is copied back, so the user's own Codex
    keeps a valid refresh token (a refresh replaces the previous one)."""
    main, mine = codex_login_file(), Path(home) / "auth.json"
    try:
        if into_home:
            Path(home).mkdir(parents=True, exist_ok=True)
            mine.write_bytes(main.read_bytes())
            os.chmod(mine, 0o600)
            return
        if not mine.exists():
            return
        a, b = json.loads(mine.read_text()), json.loads(main.read_text())
        if (a.get("last_refresh") or "") > (b.get("last_refresh") or ""):
            tmp = main.with_name(main.name + ".mechfoundry.tmp")
            tmp.write_bytes(mine.read_bytes())
            os.chmod(tmp, 0o600)
            os.replace(tmp, main)
    except (OSError, ValueError):
        pass


# ------------------------------------------------------------------------------ discovery
_OPENAI_CACHE: dict = {}
_CLAUDE_CACHE: dict = {}
# Claude Code's short names for the current model of each family (its initialize reply lists
# older ones by their full id); the menu offers these, resolved to the model they stand for.
CLAUDE_ALIASES = ("opus", "fable", "sonnet", "haiku")


def _claude_catalog() -> list[dict]:
    """The model list Claude Code reports for the user's login when the Claude Agent SDK connects
    (its initialize reply's ``models``). Connecting starts the local CLI and sends no message, so
    it costs nothing. Run on a thread of its own, so it works from inside a running event loop."""
    import asyncio
    import threading
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    async def ask():
        opts = dict(setting_sources=[], tools=[], strict_mcp_config=True)
        if claude_cli():
            opts["cli_path"] = claude_cli()
        async with ClaudeSDKClient(options=ClaudeAgentOptions(**opts)) as client:
            return list((await client.get_server_info() or {}).get("models") or [])
    out: dict = {}

    def run():
        try:
            out["models"] = asyncio.run(asyncio.wait_for(ask(), 45))
        except BaseException as ex:  # noqa: BLE001 (a missing login, a CLI that will not start)
            out["error"] = ex
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(60)
    return out.get("models") or []


def claude_models() -> list[dict]:
    """[{id, label, efforts, default_effort}] for the current Claude models of the user's login, as
    Claude Code lists them (``_claude_catalog``): each family's latest model, with the effort levels
    it takes (none for a model without an effort setting). Cached for ten minutes; the last good
    list, else ``CLAUDE_MODELS``, when Claude Code cannot be asked."""
    if _CLAUDE_CACHE.get("t", 0) > time.time() - 600:
        return _CLAUDE_CACHE["models"]
    models = []
    try:
        catalog = _claude_catalog() if claude_cli() else []
    except ImportError:
        catalog = []
    seen = set()
    for m in catalog:
        mid = m.get("resolvedModel") or m.get("value")
        if m.get("value") not in CLAUDE_ALIASES or not mid or mid in seen:
            continue
        seen.add(mid)
        efforts = list(m.get("supportedEffortLevels") or []) if m.get("supportsEffort") else []
        name = m.get("displayName") or mid
        models.append({"id": mid, "label": name if name.startswith("Claude") else f"Claude {name}",
                       "efforts": efforts, "default_effort": "high" if "high" in efforts else None})
    if not models:
        models = _CLAUDE_CACHE.get("models") or [
            {"id": mid, "label": label, "efforts": list(CLAUDE_EFFORTS), "default_effort": "high"}
            for mid, label in CLAUDE_MODELS]
    _CLAUDE_CACHE.update(t=time.time(), models=models)
    return models


def openai_models() -> list[dict]:
    """[{slug, label, efforts, default_effort}] the user's Codex login can use, from Codex's own
    catalog (``codex debug models``): the models it lists (not hidden ones) that take images, which
    the skill's visual checks need. [] without a codex CLI or a login. Cached for ten minutes."""
    cli = codex_cli()
    if not cli or not codex_login_file().exists():
        return []
    if _OPENAI_CACHE.get("t", 0) > time.time() - 600:
        return _OPENAI_CACHE["models"]
    try:
        r = subprocess.run([cli, "debug", "models"], capture_output=True, text=True, timeout=30,
                           stdin=subprocess.DEVNULL,
                           env={**os.environ, "CODEX_HOME": str(codex_login_file().parent)})
        raw = json.loads(r.stdout)
        raw = raw.get("models", raw) if isinstance(raw, dict) else raw
    except (OSError, ValueError, subprocess.SubprocessError):
        raw = []
    models = []
    for m in raw if isinstance(raw, list) else []:
        if not isinstance(m, dict) or m.get("visibility") != "list" or "image" not in (m.get("input_modalities") or []):
            continue
        efforts = [e.get("effort") for e in m.get("supported_reasoning_levels") or []
                   if e.get("effort") and e.get("effort") not in OPENAI_EFFORT_SKIP]
        models.append({"slug": m["slug"], "label": m.get("display_name") or m["slug"],
                       "efforts": efforts or list(OPEN_EFFORTS), "default_effort": m.get("default_reasoning_level"),
                       "tool_mode": m.get("tool_mode")})
    _OPENAI_CACHE.update(t=time.time(), models=models)
    return models


def parse(op: str) -> dict:
    """{agent, model, provider} for an operator id."""
    if (op or "").startswith("codex:"):
        model, _, provider = op[len("codex:"):].partition("@")
        return {"agent": "codex", "model": model, "provider": provider or "hf"}
    return {"agent": "claude", "model": op or "", "provider": None}


def operators() -> list[dict]:
    """Every operator, in menu order: {id, label, agent, provider, model, group, efforts,
    default_effort, available, unavailable (why not, or None), price (USD per M tokens, or None)}."""
    out = []
    try:
        import claude_agent_sdk  # noqa: F401
        sdk = None
    except ImportError:
        sdk = "the Claude Agent SDK is not installed (pip install claude-agent-sdk)"
    claude_why = sdk or (None if claude_cli() else "Claude Code (the `claude` command) is not installed")
    for m in (claude_models() if claude_why is None else
              [{"id": mid, "label": label, "efforts": list(CLAUDE_EFFORTS), "default_effort": "high"}
               for mid, label in CLAUDE_MODELS]):
        out.append({"id": m["id"], "label": m["label"], "agent": "claude", "provider": None, "model": m["id"],
                    "group": GROUPS["claude"], "efforts": m["efforts"], "default_effort": m["default_effort"],
                    "available": claude_why is None, "unavailable": claude_why, "price": None})
    codex = codex_cli()
    for key, m in codex_models().items():
        vendor = (m.get("vendor") or [None])[0]
        routes = [("hf", "Hugging Face", "HF_TOKEN")]
        if vendor in VENDORS:
            routes.append(("vendor",) + VENDORS[vendor])
        for provider, pname, env_key in routes:
            why = None
            if not codex:
                why = "Codex (the `codex` command) is not installed"
            elif not provider_key(env_key)[0]:
                why = f"no {env_key}"
            elif provider == "hf" and not HF_RELAY.exists():
                why = "experiments/agentic_batch lacks hf_relay.py"
            out.append({"id": f"codex:{key}@{provider}", "label": f"{m.get('label') or key} · {pname}",
                        "agent": "codex", "provider": provider, "model": key, "group": GROUPS["open"],
                        "efforts": list(OPEN_EFFORTS), "default_effort": m.get("effort"),
                        "available": why is None, "unavailable": why, "price": m.get("price")})
    for m in openai_models():
        out.append({"id": f"codex:{m['slug']}@openai", "label": m["label"], "agent": "codex",
                    "provider": "openai", "model": m["slug"], "group": GROUPS["openai"],
                    "efforts": m["efforts"], "default_effort": m["default_effort"],
                    "available": True, "unavailable": None, "price": None})
    return out


def find(op: str) -> dict | None:
    return next((o for o in operators() if o["id"] == op), None)


# ------------------------------------------------------------------- an OpenAI CODEX_HOME
def write_openai_home(home: Path, model: str, effort: str | None, notes: list[Path]) -> dict:
    """config.toml and rules/ for an OpenAI model on the user's Codex login, as
    ``ab_helper.py codex-home`` writes them for the open-weight models (approval never, the
    workspace-write sandbox, the same Codex tools off, no AGENTS.md, the developer notes, the
    runner's two command forms allowed out of the sandbox), with Codex's built-in OpenAI provider
    and the login copied in (``sync_login``). Returns the model as codex-home prints it."""
    ab = ab_helper()
    home = Path(home)
    (home / "rules").mkdir(parents=True, exist_ok=True)
    cat = next((m for m in openai_models() if m["slug"] == model), None)
    if cat and effort not in cat["efforts"]:
        effort = cat["default_effort"]
    note = "\n\n".join(Path(f).read_text(encoding="utf-8").strip() for f in notes if f)
    cfg = ['model_provider = "openai"', f"model = {ab._toml(model)}", 'approval_policy = "never"',
           'sandbox_mode = "workspace-write"', 'web_search = "disabled"', "project_doc_max_bytes = 0"]
    if effort:
        cfg.append(f"model_reasoning_effort = {ab._toml(effort)}")
    if note:
        cfg.append(f"developer_instructions = {ab._toml(note)}")
    # OpenAI's models are `code_mode_only` in Codex's catalog: every tool call goes through the
    # code-mode host (codex-code-mode-host, shipped beside codex). The open-weight runs turn the
    # host off; with it off these models have no tools at all ("Code Mode is unavailable because
    # code-mode host is disabled ... fail closed"), so it stays on for them.
    keep = {"code_mode_host"} if (cat or {}).get("tool_mode", "code_mode_only") == "code_mode_only" else set()
    cfg += ["", "[features]"] + [f"{f} = false" for f in ab.CODEX_FEATURES_OFF if f not in keep]
    (home / "config.toml").write_text("\n".join(cfg) + "\n", encoding="utf-8")
    rules = [f'prefix_rule(pattern=["python3", "-u", "{ab.OPERATOR_SCRIPTS}{s}"], decision="allow")'
             for s in ("drive.py", "inspect_session.py")]
    rules += ['prefix_rule(pattern=["git"], decision="forbidden")']
    rules += [f'prefix_rule(pattern=["{py}", "{flag}"], decision="forbidden")'
              for py in ("python3", "python") for flag in ("-", "-c")]
    (home / "rules" / "operator.rules").write_text("\n".join(rules) + "\n", encoding="utf-8")
    sync_login(home, into_home=True)
    return {"model": model, "key": model, "label": cat["label"] if cat else model, "route": None,
            "provider": "openai", "price": None, "effort": effort, "name": "OpenAI (Codex login)",
            "base_url": None, "env_key": None}


# ------------------------------------------------------------------------- the HF relay
def relay_python() -> str:
    """An interpreter with Pillow for hf_relay.py (it re-encodes pictures to stay under the router's
    5 MiB limit; without Pillow it sends them as they are): HF_RELAY_PYTHON, else the first of this
    interpreter, python3 on the PATH, conda's base python3 and /usr/bin/python3 that imports PIL."""
    import sys
    if os.environ.get("HF_RELAY_PYTHON"):
        return os.environ["HF_RELAY_PYTHON"]
    cands = [sys.executable, shutil.which("python3")]
    for exe in (os.environ.get("CONDA_EXE"), shutil.which("conda")):
        if exe:
            cands.append(str(Path(os.path.realpath(exe)).parent.parent / "bin" / "python3"))
    cands += [os.path.expanduser("~/anaconda3/bin/python3"), os.path.expanduser("~/miniconda3/bin/python3"),
              "/usr/bin/python3"]
    for c in cands:
        if c and os.path.exists(c) and subprocess.run([c, "-c", "import PIL"], capture_output=True).returncode == 0:
            return c
    return sys.executable


def relay_command(port_file: Path, log: Path) -> list[str]:
    """The command that starts hf_relay.py (ab_common.sh start_hf_relay); it writes its port to
    ``port_file`` once listening."""
    return [relay_python(), str(HF_RELAY), "--upstream", HF_UPSTREAM, "--port-file", str(port_file),
            "--log", str(log)]


# ------------------------------------------------------------------------ Codex's output
def unwrap(command: str) -> str:
    """The command the agent typed: Codex reports it wrapped as ``/bin/bash -lc '<command>'``."""
    try:
        parts = shlex.split(command)
    except ValueError:
        return command
    if len(parts) == 3 and parts[0].endswith("sh") and parts[1] in ("-lc", "-c"):
        return parts[2]
    return command


PATCH_FILE = re.compile(r"^\s*\*\*\* (Add|Update|Delete) File:\s*(.+?)\s*$")


def patch_changes(text: str) -> list[dict]:
    """[{path, kind}] a patch in apply_patch's format names (``*** Add File: PATH`` and its
    Update/Delete siblings); kind is add, update or delete, as Codex's file_change reports them."""
    out = []
    for line in (text or "").splitlines():
        m = PATCH_FILE.match(line)
        if m:
            out.append({"path": m.group(2), "kind": m.group(1).lower()})
    return out


def file_change_tool(changes: list[dict]) -> tuple[str, dict]:
    """A Codex file change (its file_change item, or a shell apply_patch) as Claude's tool would
    show it: Write when it only adds files, else Edit, with the files it changes. Codex reports an
    ``apply_patch`` heredoc typed in the shell as a file_change too."""
    changes = [{"path": str(c.get("path", "")), "kind": str(c.get("kind", ""))} for c in changes or []]
    name = "Write" if changes and all(c["kind"] == "add" for c in changes) else "Edit"
    return name, {"file_path": ", ".join(c["path"] for c in changes), "changes": changes}


def classify(command: str) -> tuple[str, dict]:
    """A Codex shell command as Claude's tool would show it: a file read (``sed -n``, ``cat``,
    ``head``, ``tail``, ``nl`` of one file) as Read, ``ls`` as Glob, ``grep``/``rg`` as Grep,
    ``apply_patch`` run in the shell (the tool note's heredoc form, for models without the
    apply_patch tool) as Write when it only adds files, else Edit, with the files it names;
    anything else (a pipeline, a runner) as Bash with the command as typed."""
    inner = unwrap(command)
    if inner.lstrip().startswith("apply_patch"):
        changes = patch_changes(inner)
        if changes:
            name, tool_input = file_change_tool(changes)
            return name, {**tool_input, "command": inner}
    try:
        argv = shlex.split(inner)
    except ValueError:
        return "Bash", {"command": inner}
    if not argv or any(t in ("|", "&&", "||", ";", ">", ">>", "<") for t in argv):
        return "Bash", {"command": inner}
    head, args = argv[0], argv[1:]
    files = [a for a in args if not a.startswith("-")]
    if head == "sed" and args[:1] == ["-n"] and len(files) == 2:
        return "Read", {"file_path": files[1], "lines": files[0].rstrip("p")}
    if head in ("cat", "head", "tail", "nl") and len(files) == 1:
        return "Read", {"file_path": files[0]}
    if head == "ls":
        return "Glob", {"pattern": " ".join(files) or "."}
    if head in ("grep", "rg", "egrep") and files:
        return "Grep", {"pattern": files[0], "path": " ".join(files[1:])}
    return "Bash", {"command": inner}


def short_error(message: str, limit: int = 300) -> str:
    """A provider's error in a line: its innermost "message" when it is JSON, cut to ``limit``."""
    found = re.findall(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', message or "")
    text = (found[-1] if found else (message or "")).replace('\\"', '"')
    return text if len(text) <= limit else text[:limit] + "…"
