#!/usr/bin/env python3
"""A local translator between the Codex CLI and Hugging Face's router (run_batch.sh --agent codex,
provider hf): Codex speaks only the Responses API; this serves it on 127.0.0.1 and calls the
router's Chat Completions endpoint.

    python3 hf_relay.py --port-file FILE [--upstream https://router.huggingface.co/v1] [--log FILE]

Why not the router's own Responses endpoint (`/v1/responses`, Hugging Face's responses.js, which
does the same translation server-side).  Measured 2026-11-17 with codex-cli 0.163.0:
  1. it refuses any request body over 100 KB (102 400 bytes, Express's default JSON limit; 413
     Payload Too Large, gzip or not), and Codex resends the whole conversation each turn, so an
     operator session dies within its first minute;
  2. it drops every `developer` message (src/messages.ts converts only system, user and
     assistant), and Codex sends its whole system prompt -- and the runner's notes -- as developer
     messages, so the model ran without any;
  3. it refuses a tool result that carries an image (Codex's view_image; 400 invalid_union,
     "Expected string, received array").
The router's Chat Completions endpoint takes bodies of several MB (a 2.2 MB request was accepted).

Request: POST .../responses -> POST <upstream>/chat/completions, as responses.js maps it
(github.com/huggingface/responses.js, src/messages.ts and src/routes/responses.ts, read 2026-11-17):
  - messages: `system`, `user` and `assistant` items as they are, and `developer` items as `system`
    (2); content parts input_text/output_text -> text, input_image -> image_url;
  - `function_call` -> an assistant message's tool_calls; `function_call_output` -> a `tool` message;
    a `reasoning` item -> the next assistant message's `reasoning_content` (its verbatim text, else its
    summary); consecutive assistant items merge into one message;
  - a tool result whose output is a list (view_image) becomes a string, and its images go in a user
    message after the run of tool messages it belongs to (3): "Image returned by view_image (call
    <id>):" + the image;
  - function tools -> chat tools; tool_choice, parallel_tool_calls, max_output_tokens (max_tokens),
    reasoning.effort (reasoning_effort), temperature and top_p are passed on; Codex's store,
    include, prompt_cache_key, client_metadata (it holds the repository's git remote and paths) and
    reasoning.summary are not.
Response: the chat stream becomes the Responses events responses.js emits (data-only SSE):
response.created / in_progress; per output item output_item.added, the deltas (reasoning_text,
output_text, function_call_arguments) and their .done events, output_item.done; response.completed
with the usage (prompt -> input tokens with the cached part, completion -> output tokens with the
reasoning part).  An upstream error before the stream is returned with its status, as JSON; one
inside the stream ends it with response.failed.  Every other request (GET /models, ...) is passed
through unchanged.

Images (the router's Chat Completions endpoint refuses bodies over 5 MiB: 5 084 484 bytes passed and
5 466 204 were refused in a 2026-11-17 session, and Codex resends every picture of the session as a
PNG data URL each turn -- 16 pictures came to 5.0 MB).  Every data-URL image is re-encoded as a JPEG
(quality 90, transparency onto white), its longer side capped at 1568 px -- the size Anthropic's API
also scales larger images down to, so the Claude runs saw no more detail than this -- whenever that
makes it smaller; each image is converted once and cached.  A picture identical to one sent earlier
in the request (the agent opened the same file again) is sent once: the later copy becomes a line
naming the earlier one -- the first copy is kept, not the latest, so the earlier messages, and with
them the provider's prompt cache, stay unchanged.  If a request is still over
HF_RELAY_MAX_BYTES (default 4 900 000), the oldest images are replaced by a line saying so, until it
fits; the log counts them.  Without Pillow the images go as they are (logged at start).

Logging: one line per request -- time, path, status, request and reply bytes, messages, images
moved, output items, tokens.  Never a header, never a body.  HF_RELAY_DUMP=<dir> (debugging) writes
each Responses request body as received to <dir>.
"""
import argparse
import base64
import hashlib
import http.client
import io
import json
import os
import secrets
import socket
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding", "content-encoding"}
TOOL_RESULTS = ("function_call_output", "custom_tool_call_output")
MAX_SIDE = 1568
MAX_BYTES = int(os.environ.get("HF_RELAY_MAX_BYTES") or 4_900_000)
try:
    from PIL import Image
except ImportError:          # the relay still works; the images are not re-encoded
    Image = None
_SHRUNK = {}                 # data URL -> its re-encoded form (Codex resends each image every turn)
_SHRUNK_LOCK = threading.Lock()


def shrink(url):
    """A data-URL image as a JPEG of at most MAX_SIDE px, when that is smaller; else the URL."""
    if Image is None or not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
        return url
    key = hashlib.sha1(url.encode()).hexdigest()
    with _SHRUNK_LOCK:
        if key in _SHRUNK:
            return _SHRUNK[key]
    try:
        img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        img.load()
        resized = max(img.size) > MAX_SIDE
        if resized:
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            flat = Image.new("RGB", img.size, (255, 255, 255))
            flat.paste(img, mask=img.split()[-1])
            img = flat
        elif img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        new = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        out = new if (resized or len(new) < len(url)) else url
    except Exception:        # an image Pillow cannot read goes as it is
        out = url
    with _SHRUNK_LOCK:
        _SHRUNK[key] = out
    return out


def dedupe(chat):
    """Send each identical image once: a later copy becomes a note naming the first; returns how many."""
    seen, repeats = {}, 0
    for i, m in enumerate(chat["messages"]):
        if not isinstance(m.get("content"), list):
            continue
        label = None
        for k, part in enumerate(m["content"]):
            if part.get("type") == "text" and part.get("text", "").startswith("Image returned by "):
                label = part["text"][len("Image returned by "):].rstrip(":")
            if part.get("type") != "image_url":
                continue
            key = hashlib.sha1(part["image_url"]["url"].encode()).hexdigest()
            if key in seen:
                m["content"][k] = {"type": "text", "text": f"[The same image as the one returned earlier by {seen[key]}; "
                                   "it is shown once, there.]"}
                repeats += 1
            else:
                seen[key] = label or "an earlier message"
    return repeats


def fit(chat):
    """Replace the oldest images by a note until the request is under MAX_BYTES; returns how many."""
    size = len(json.dumps(chat))
    dropped = 0
    if size <= MAX_BYTES:
        return 0
    for m in chat["messages"]:
        if not isinstance(m.get("content"), list):
            continue
        for k, part in enumerate(m["content"]):
            if part.get("type") == "image_url":
                url = part["image_url"]["url"]
                m["content"][k] = {"type": "text", "text": "[An earlier image was left out here to keep the request "
                                   "under the provider's size limit; view it again if you need it.]"}
                size -= len(url)
                dropped += 1
                if size <= MAX_BYTES:
                    return dropped
    return dropped


def uid(prefix):
    return f"{prefix}_{secrets.token_hex(24)}"


# ---------------------------------------------------------------- request: Responses -> Chat

def _parts(content):
    """A Responses message's content -> chat content (string, or a list of text/image_url parts)."""
    if isinstance(content, str):
        return content
    out = []
    for c in content or []:
        if not isinstance(c, dict):
            continue
        t = c.get("type")
        if t in ("input_text", "output_text", "text") and c.get("text"):
            out.append({"type": "text", "text": c["text"]})
        elif t == "input_image":
            url = shrink(c.get("image_url")) or (c.get("file_id") and f"file:{c['file_id']}")
            if url:
                img = {"url": url}
                if c.get("detail") and c["detail"] != "auto":
                    img["detail"] = c["detail"]
                out.append({"type": "image_url", "image_url": img})
    return out


def _tool_output(item, names, pending):
    """A tool result's output as a string; its images are appended to `pending` (a user message)."""
    output = item.get("output")
    if isinstance(output, str):
        return output, 0
    texts, images, others = [], [], []
    for p in output or []:
        if not isinstance(p, dict):
            continue
        if p.get("type") in ("input_text", "output_text", "text"):
            texts.append(p.get("text") or "")
        elif p.get("type") == "input_image":
            images.append(p)
        else:
            others.append(p)
    text = "\n".join(t for t in texts if t)
    if others:
        text += ("\n" if text else "") + json.dumps(others)
    if images:
        plural = len(images) > 1
        text += ("\n" if text else "") + f"(The image{'s' if plural else ''} follow{'' if plural else 's'} in the next message.)"
        pending.append({"type": "text", "text": f"Image returned by {names.get(item.get('call_id')) or 'the tool'} "
                                                f"(call {item.get('call_id')}):"})
        pending.extend(_parts(images))
    return text, len(images)


def to_chat(body):
    """(chat request, number of images moved) for one Responses request body."""
    messages = []
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})
    items = body.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    items = items or []
    names = {it.get("call_id"): it.get("name") for it in items if isinstance(it, dict)
             and it.get("type") in ("function_call", "custom_tool_call")}
    pending, moved = [], 0

    def flush_images():
        nonlocal pending
        if pending:
            messages.append({"role": "user", "content": pending})
            pending = []

    def assistant():
        """The assistant message to add to: the last one, if it is one."""
        if messages and messages[-1]["role"] == "assistant" and not pending:
            return messages[-1]
        flush_images()
        messages.append({"role": "assistant", "content": []})
        return messages[-1]

    for it in items:
        if not isinstance(it, dict):
            continue
        t = it.get("type") or "message"
        if t not in TOOL_RESULTS:
            flush_images()
        if t == "message":
            role = it.get("role")
            content = _parts(it.get("content"))
            if role == "assistant":
                a = assistant()
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}] if content else []
                a["content"] += content
            elif role in ("user", "system", "developer"):
                if content:
                    messages.append({"role": "system" if role == "developer" else role, "content": content})
        elif t == "reasoning":
            parts = it.get("content") or it.get("summary") or []
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
            if text:
                a = assistant()
                a["reasoning_content"] = a.get("reasoning_content", "") + text
        elif t in ("function_call", "custom_tool_call"):
            args = it.get("arguments") if t == "function_call" else json.dumps({"input": it.get("input", "")})
            assistant().setdefault("tool_calls", []).append(
                {"id": it.get("call_id"), "type": "function", "function": {"name": it.get("name"), "arguments": args or ""}})
        elif t in TOOL_RESULTS:
            text, n = _tool_output(it, names, pending)
            moved += n
            # a tool message goes right after its assistant turn, before any images already pending
            messages.append({"role": "tool", "tool_call_id": it.get("call_id"), "content": text})
    flush_images()
    for m in messages:                      # as responses.js: no empty content, a lone text part as a string
        if m["role"] == "assistant":
            if not m["content"]:
                if m.get("tool_calls") or m.get("reasoning_content"):
                    del m["content"]
                else:
                    m["content"] = ""
            elif len(m["content"]) == 1 and m["content"][0]["type"] == "text":
                m["content"] = m["content"][0]["text"]
        elif isinstance(m["content"], list) and len(m["content"]) == 1 and m["content"][0]["type"] == "text":
            m["content"] = m["content"][0]["text"]
    chat = {"model": body.get("model"), "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}}
    tools = [{"type": "function", "function": {k: v for k, v in (("name", t.get("name")), ("description", t.get("description")),
                                                                 ("parameters", t.get("parameters")), ("strict", t.get("strict")))
                                               if v is not None}}
             for t in body.get("tools") or [] if isinstance(t, dict) and t.get("type") == "function"]
    if tools:
        chat["tools"] = tools
        tc = body.get("tool_choice")
        if isinstance(tc, str):
            chat["tool_choice"] = tc
        elif isinstance(tc, dict) and tc.get("name"):
            chat["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
        if "parallel_tool_calls" in body:
            chat["parallel_tool_calls"] = body["parallel_tool_calls"]
    if body.get("max_output_tokens"):
        chat["max_tokens"] = body["max_output_tokens"]
    effort = (body.get("reasoning") or {}).get("effort")
    if effort:
        chat["reasoning_effort"] = effort
    for k in ("temperature", "top_p"):
        if body.get(k) is not None:
            chat[k] = body[k]
    return chat, moved


# ---------------------------------------------------------------- response: Chat stream -> Responses events

class Stream:
    """Turns chat completion chunks into Responses stream events (dicts), as responses.js does."""

    def __init__(self, model):
        self.seq = 0
        self.response = {"id": uid("resp"), "object": "response", "created_at": int(time.time()), "model": model,
                         "status": "in_progress", "output": [], "error": None, "instructions": None,
                         "max_output_tokens": None, "metadata": None, "tool_choice": "auto", "tools": [],
                         "usage": None}
        self.open = None            # the open text/reasoning item: (item, index)
        self.calls = {}             # chat tool-call index -> (item, output index)
        self.usage = None

    def ev(self, kind, **kw):
        self.seq += 1
        return {"type": kind, **kw, "sequence_number": self.seq}

    def start(self):
        return [self.ev("response.created", response=dict(self.response)),
                self.ev("response.in_progress", response=dict(self.response))]

    def _close_open(self):
        out = []
        if self.open:
            item, idx = self.open
            part = item["content"][0]
            if item["type"] == "message":
                out += [self.ev("response.output_text.done", item_id=item["id"], output_index=idx, content_index=0, text=part["text"]),
                        self.ev("response.content_part.done", item_id=item["id"], output_index=idx, content_index=0, part=part)]
            else:
                out += [self.ev("response.reasoning_text.done", item_id=item["id"], output_index=idx, content_index=0, text=part["text"])]
            item["status"] = "completed"
            out.append(self.ev("response.output_item.done", output_index=idx, item=item))
            self.open = None
        return out

    def _text(self, kind, text):
        out = []
        if not self.open or self.open[0]["type"] != kind:
            out += self._close_open()
            if kind == "message":
                item = {"id": uid("msg"), "type": "message", "role": "assistant", "status": "in_progress", "content": []}
            else:
                item = {"id": uid("rs"), "type": "reasoning", "summary": [], "content": [], "encrypted_content": None}
            self.response["output"].append(item)
            idx = len(self.response["output"]) - 1
            out.append(self.ev("response.output_item.added", output_index=idx, item={**item, "content": []}))
            part = {"type": "output_text", "text": "", "annotations": []} if kind == "message" else {"type": "reasoning_text", "text": ""}
            item["content"].append(part)
            if kind == "message":
                out.append(self.ev("response.content_part.added", item_id=item["id"], output_index=idx, content_index=0,
                                   part=dict(part)))
            self.open = (item, idx)
        item, idx = self.open
        item["content"][0]["text"] += text
        delta = "response.output_text.delta" if kind == "message" else "response.reasoning_text.delta"
        out.append(self.ev(delta, item_id=item["id"], output_index=idx, content_index=0, delta=text))
        return out

    def chunk(self, c):
        out = []
        if isinstance(c.get("usage"), dict):
            self.usage = c["usage"]
        for ch in c.get("choices") or []:
            d = ch.get("delta") or {}
            reasoning = d.get("reasoning") or d.get("reasoning_content")
            if reasoning:
                out += self._text("reasoning", reasoning)
            if d.get("content"):
                out += self._text("message", d["content"])
            for tc in d.get("tool_calls") or []:
                k = tc.get("index", 0)
                fn = tc.get("function") or {}
                if k not in self.calls:
                    out += self._close_open()
                    item = {"type": "function_call", "id": uid("fc"), "call_id": tc.get("id") or uid("call"),
                            "name": fn.get("name") or "", "arguments": "", "status": "in_progress"}
                    self.response["output"].append(item)
                    self.calls[k] = (item, len(self.response["output"]) - 1)
                    out.append(self.ev("response.output_item.added", output_index=self.calls[k][1], item=dict(item)))
                item, idx = self.calls[k]
                if fn.get("name") and not item["name"]:
                    item["name"] = fn["name"]
                if fn.get("arguments"):
                    item["arguments"] += fn["arguments"]
                    out.append(self.ev("response.function_call_arguments.delta", item_id=item["id"], output_index=idx,
                                       delta=fn["arguments"]))
        return out

    def finish(self):
        out = self._close_open()
        for k in sorted(self.calls):
            item, idx = self.calls[k]
            item["status"] = "completed"
            out += [self.ev("response.function_call_arguments.done", item_id=item["id"], output_index=idx,
                            arguments=item["arguments"]),
                    self.ev("response.output_item.done", output_index=idx, item=item)]
        u = self.usage or {}
        prompt, completion = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        self.response["usage"] = {
            "input_tokens": prompt,
            "input_tokens_details": {"cached_tokens": int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)},
            "output_tokens": completion,
            "output_tokens_details": {"reasoning_tokens": int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)},
            "total_tokens": int(u.get("total_tokens") or prompt + completion)}
        self.response["status"] = "completed"
        out.append(self.ev("response.completed", response=self.response))
        return out

    def fail(self, message, code="server_error"):
        self.response["status"] = "failed"
        self.response["error"] = {"code": code, "message": message}
        return [self.ev("response.failed", response=self.response)]


# ---------------------------------------------------------------- the server

class Relay(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = None
    log_path = None
    lock = threading.Lock()

    def log_message(self, *args):        # the access log is ours (note()), not BaseHTTPRequestHandler's
        pass

    def note(self, line):
        if self.log_path:
            with self.lock, open(self.log_path, "a") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")

    def _upstream(self, method, path, data, headers):
        up = urllib.parse.urlsplit(self.upstream)
        conn = http.client.HTTPSConnection(up.hostname, up.port or 443, timeout=900)
        headers = {**headers, "Content-Length": str(len(data))}
        conn.request(method, up.path.rstrip("/") + path, body=data if data else None, headers=headers)
        return conn, conn.getresponse()

    def _reply(self, status, body, ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _headers(self):
        return {k: v for k, v in self.headers.items() if k.lower() not in HOP}

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(length) if length else b""
        sub = self.path[len("/v1"):] if self.path.startswith("/v1") else self.path
        if sub.split("?")[0].rstrip("/") == "/responses" and not self.headers.get("Content-Encoding"):
            try:
                body = json.loads(data)
            except (ValueError, UnicodeDecodeError):
                body = None
            if isinstance(body, dict):
                return self._responses(body, data)
        self._passthrough(data)

    do_GET = lambda self: self._passthrough(b"")  # noqa: E731

    def _passthrough(self, data):
        sub = self.path[len("/v1"):] if self.path.startswith("/v1") else self.path
        try:
            conn, resp = self._upstream(self.command, sub, data, self._headers())
        except (OSError, http.client.HTTPException) as e:
            self.note(f"{self.command} {self.path} upstream error {type(e).__name__}")
            return self._reply(502, json.dumps({"error": {"message": f"hf_relay: upstream unreachable: {e}"}}).encode())
        body = resp.read()
        conn.close()
        self.note(f"{self.command} {self.path} passthrough {resp.status} sent={len(body)}")
        self._reply(resp.status, body, resp.getheader("Content-Type") or "application/json")

    def _responses(self, body, raw):
        dump = os.environ.get("HF_RELAY_DUMP")       # debugging only: every request body, as received
        if dump:
            os.makedirs(dump, exist_ok=True)
            with open(os.path.join(dump, f"{time.time():.6f}.json"), "wb") as fh:
                fh.write(raw)
        chat, moved = to_chat(body)
        repeats = dedupe(chat)
        dropped = fit(chat)
        data = json.dumps(chat).encode()
        headers = {k: v for k, v in self._headers().items() if k.lower() in ("authorization", "user-agent", "x-hf-bill-to")}
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"
        try:
            conn, resp = self._upstream("POST", "/chat/completions", data, headers)
        except (OSError, http.client.HTTPException) as e:
            self.note(f"POST {self.path} upstream error {type(e).__name__} req={len(data)}")
            return self._reply(502, json.dumps({"error": {"message": f"hf_relay: upstream unreachable: {e}"}}).encode())
        if resp.status != 200:
            err = resp.read()
            conn.close()
            self.note(f"POST {self.path} -> chat {resp.status} req={len(data)} messages={len(chat['messages'])}")
            try:
                json.loads(err)
                out = err
            except ValueError:
                out = json.dumps({"error": {"message": err[:2000].decode("utf-8", "replace"), "code": resp.status}}).encode()
            return self._reply(resp.status, out)
        stream = Stream(body.get("model"))
        streaming = body.get("stream", True)
        sent = 0
        if streaming:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

        def emit(events):
            nonlocal sent
            if not streaming:
                return
            for e in events:
                chunk = f"data: {json.dumps(e)}\n\n".encode()
                self.wfile.write(chunk)
                sent += len(chunk)
            self.wfile.flush()

        failed = None
        try:
            emit(stream.start())
            buf = b""
            while True:
                piece = resp.read1(65536)
                if not piece:
                    break
                buf += piece
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        continue
                    try:
                        c = json.loads(payload)
                    except ValueError:
                        continue
                    if c.get("error"):
                        failed = c["error"].get("message") if isinstance(c["error"], dict) else str(c["error"])
                        break
                    emit(stream.chunk(c))
                if failed:
                    break
            emit(stream.fail(failed) if failed else stream.finish())
        except (BrokenPipeError, ConnectionResetError, socket.timeout) as e:
            failed = failed or type(e).__name__
        finally:
            conn.close()
        if not streaming:
            if not failed:
                stream.finish()
            return self._reply(200 if not failed else 502, json.dumps(stream.response).encode())
        self.close_connection = True
        u = stream.response.get("usage") or {}
        self.note(f"POST {self.path} -> chat 200 req={len(data)} messages={len(chat['messages'])} images_moved={moved} "
                  f"images_repeated={repeats} images_dropped={dropped} "
                  f"items={len(stream.response['output'])} in={u.get('input_tokens')} cached="
                  f"{(u.get('input_tokens_details') or {}).get('cached_tokens')} out={u.get('output_tokens')} sent={sent}"
                  + (f" FAILED {str(failed)[:200]!r}" if failed else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="https://router.huggingface.co/v1")
    ap.add_argument("--port-file", required=True, help="written with the port once listening")
    ap.add_argument("--log", default="")
    a = ap.parse_args()
    Relay.upstream, Relay.log_path = a.upstream, a.log or None
    if Image is None and Relay.log_path:
        with open(Relay.log_path, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} no Pillow in {sys.executable}: images are sent as they are\n")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Relay)
    server.daemon_threads = True
    with open(a.port_file + ".tmp", "w") as fh:
        fh.write(str(server.server_address[1]))
    os.replace(a.port_file + ".tmp", a.port_file)
    server.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
