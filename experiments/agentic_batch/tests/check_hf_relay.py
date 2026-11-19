"""hf_relay: the Responses -> Chat request conversion and the chat stream -> Responses events.
python3 tests/check_hf_relay.py [CODEX_REQUEST.json ...]  (optional: real request bodies dumped with HF_RELAY_DUMP)"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hf_relay import Stream, to_chat  # noqa: E402

IMG = {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "high"}
dev = {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "system prompt"}]}
user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]}
call = lambda cid, name, args="{}": {"type": "function_call", "name": name, "arguments": args, "call_id": cid}  # noqa: E731
text_out = lambda cid: {"type": "function_call_output", "call_id": cid, "output": "ok"}  # noqa: E731
img_out = lambda cid: {"type": "function_call_output", "call_id": cid, "output": [IMG]}  # noqa: E731
reason = lambda t: {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": t}]}  # noqa: E731
say = lambda t: {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": t}]}  # noqa: E731
TOOLS = [{"type": "function", "name": "exec_command", "description": "run", "strict": False,
          "parameters": {"type": "object", "properties": {}}}]

# Developer messages become system (responses.js drops them); Codex-only fields are not sent.
chat, moved = to_chat({"model": "m:route", "input": [dev, user], "tools": TOOLS, "tool_choice": "auto",
                       "parallel_tool_calls": True, "store": False, "include": ["x"], "prompt_cache_key": "k",
                       "client_metadata": {"repo": "secret"}, "reasoning": {"summary": "auto"}, "stream": True})
assert chat["messages"] == [{"role": "system", "content": "system prompt"}, {"role": "user", "content": "go"}], chat
assert chat["model"] == "m:route" and chat["stream"] and chat["stream_options"] == {"include_usage": True}
assert chat["tools"] == [{"type": "function", "function": {"name": "exec_command", "description": "run",
                                                          "parameters": {"type": "object", "properties": {}}, "strict": False}}]
assert chat["tool_choice"] == "auto" and chat["parallel_tool_calls"] is True
assert not {"store", "include", "prompt_cache_key", "client_metadata", "reasoning"} & set(chat) and moved == 0

# One assistant turn: reasoning + text + two parallel calls merge into one message; then the tool
# messages; the image of the second result in a user message after both.
chat, moved = to_chat({"input": [user, reason("think"), say("ok"), call("a", "exec_command", '{"cmd":"ls"}'),
                                 call("v", "view_image"), text_out("a"), img_out("v"), reason("next"), say("done")]})
m = chat["messages"]
assert [x["role"] for x in m] == ["user", "assistant", "tool", "tool", "user", "assistant"], [x["role"] for x in m]
assert m[1]["reasoning_content"] == "think" and m[1]["content"] == "ok"
assert [t["id"] for t in m[1]["tool_calls"]] == ["a", "v"] and m[1]["tool_calls"][0]["function"] == {"name": "exec_command", "arguments": '{"cmd":"ls"}'}
assert m[2] == {"role": "tool", "tool_call_id": "a", "content": "ok"}
assert m[3] == {"role": "tool", "tool_call_id": "v", "content": "(The image follows in the next message.)"}
assert m[4]["content"][0] == {"type": "text", "text": "Image returned by view_image (call v):"}
assert m[4]["content"][1] == {"type": "image_url", "image_url": {"url": IMG["image_url"], "detail": "high"}} and moved == 1
assert m[5] == {"role": "assistant", "reasoning_content": "next", "content": "done"}
# An assistant turn with only tool calls has no content key.
chat, _ = to_chat({"input": [user, call("a", "exec_command")]})
assert "content" not in chat["messages"][1]

# The stream: reasoning, text, then a tool call in pieces, then usage.
s = Stream("m")
ev = s.start()
for c in ({"choices": [{"delta": {"reasoning_content": "hm"}}]}, {"choices": [{"delta": {"reasoning_content": "m"}}]},
          {"choices": [{"delta": {"content": "Hi"}}]},
          {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "exec_command", "arguments": "{\"cm"}}]}}]},
          {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "d\":1}"}}]}}]},
          {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                                    "prompt_tokens_details": {"cached_tokens": 60}, "completion_tokens_details": {"reasoning_tokens": 5}}}):
    ev += s.chunk(c)
ev += s.finish()
kinds = [e["type"] for e in ev]
assert kinds[:2] == ["response.created", "response.in_progress"] and kinds[-1] == "response.completed", kinds
assert [e["sequence_number"] for e in ev] == list(range(1, len(ev) + 1))
done = [e["item"] for e in ev if e["type"] == "response.output_item.done"]
assert [d["type"] for d in done] == ["reasoning", "message", "function_call"], done
assert done[0]["content"] == [{"type": "reasoning_text", "text": "hmm"}]
assert done[1]["content"][0]["text"] == "Hi" and done[1]["status"] == "completed"
assert done[2]["call_id"] == "call_1" and done[2]["name"] == "exec_command" and done[2]["arguments"] == '{"cmd":1}'
final = ev[-1]["response"]
assert final["status"] == "completed" and [o["type"] for o in final["output"]] == ["reasoning", "message", "function_call"]
assert final["usage"] == {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 60}, "output_tokens": 20,
                          "output_tokens_details": {"reasoning_tokens": 5}, "total_tokens": 120}
assert Stream("m").fail("boom")[-1]["response"]["error"]["message"] == "boom"

# Real Codex request bodies, if given: every item is converted and nothing Codex-only leaks through.
for path in sys.argv[1:]:
    body = json.load(open(path))
    chat, moved = to_chat(body)
    roles = [x["role"] for x in chat["messages"]]
    assert "client_metadata" not in json.dumps(chat) and roles and set(roles) <= {"system", "user", "assistant", "tool"}
    n_calls = sum(1 for i in body["input"] if i.get("type") == "function_call")
    assert sum(len(x.get("tool_calls", [])) for x in chat["messages"]) == n_calls
    assert roles.count("tool") == sum(1 for i in body["input"] if i.get("type") == "function_call_output")
    print(f"  {os.path.basename(path)}: {len(body['input'])} items -> {len(roles)} messages {roles[:6]}...")
print("hf_relay conversion: ALL PASS")

# Images: a large PNG becomes a smaller JPEG at most 1568 px; a tiny one is left as it is; the size
# guard replaces the oldest images first.
import base64, io  # noqa: E401,E402
import hf_relay  # noqa: E402
if hf_relay.Image is not None:
    def png(w, h, noise=True):
        img = hf_relay.Image.effect_noise((w, h), 60).convert("RGB") if noise else hf_relay.Image.new("RGB", (w, h), (200, 30, 30))
        buf = io.BytesIO(); img.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    big = png(2000, 1400)
    out = hf_relay.shrink(big)
    assert out.startswith("data:image/jpeg;base64,") and len(out) < len(big)
    assert max(hf_relay.Image.open(io.BytesIO(base64.b64decode(out.split(",", 1)[1]))).size) == hf_relay.MAX_SIDE
    small = png(8, 8, noise=False)
    assert hf_relay.shrink(small) in (small, hf_relay.shrink(small))
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x" * 3000}}]} for _ in range(3)]
    old, hf_relay.MAX_BYTES = hf_relay.MAX_BYTES, 4000   # three 3000-char images: dropping two fits
    chat = {"messages": msgs}
    assert hf_relay.fit(chat) == 2 and chat["messages"][0]["content"][0]["type"] == "text" \
        and chat["messages"][2]["content"][0]["type"] == "image_url"
    hf_relay.MAX_BYTES = old
    print("hf_relay images: ALL PASS")

# A repeated image is sent once: the first copy stays, a later identical one becomes a note naming it;
# a different image is kept.
A, B = {"type": "input_image", "image_url": "data:image/png;base64,QUFB"}, {"type": "input_image", "image_url": "data:image/png;base64,QkJC"}
chat, _ = to_chat({"input": [user, call("v1", "view_image"), {"type": "function_call_output", "call_id": "v1", "output": [A]},
                             call("v2", "view_image"), {"type": "function_call_output", "call_id": "v2", "output": [B]},
                             call("v3", "view_image"), {"type": "function_call_output", "call_id": "v3", "output": [A]}]})
first = json.dumps(chat["messages"][:5])
assert hf_relay.dedupe(chat) == 1
assert json.dumps(chat["messages"][:5]) == first           # earlier messages untouched (the prompt cache holds)
last = chat["messages"][-1]["content"]
assert last[0]["text"] == "Image returned by view_image (call v3):" and last[1]["type"] == "text" \
    and "returned earlier by view_image (call v1)" in last[1]["text"], last
imgs = [p for m in chat["messages"] if isinstance(m.get("content"), list) for p in m["content"] if p["type"] == "image_url"]
assert len(imgs) == 2
print("hf_relay repeated images: ALL PASS")
