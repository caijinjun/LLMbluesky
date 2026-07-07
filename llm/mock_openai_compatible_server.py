"""Tiny local OpenAI-compatible mock server for the ATC HMI demo.

Run:
    python llm/mock_openai_compatible_server.py --host 127.0.0.1 --port 8000

Then set:
    ATC_LLM_API_URL=http://127.0.0.1:8000/v1/chat/completions
    ATC_LLM_MODEL=mock-atc-explainer

This server is intentionally deterministic. It verifies the GUI's LLM API
boundary without requiring network access or a real model service.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def build_response_text(payload: dict) -> str:
    messages = payload.get("messages", [])
    user_text = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_text = str(msg.get("content", ""))
    marker = "JSON input:\n"
    plan = {}
    if marker in user_text:
        raw = user_text.split(marker, 1)[1].strip()
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            plan = {}

    instructions = plan.get("standard_instructions") or []
    conflicts = plan.get("conflicts") or []
    preference = plan.get("preference", "unknown")
    conflict_text = ", ".join("/".join(item.get("aircraft", [])) for item in conflicts) or "current conflict set"
    instruction_text = " ".join(instructions) or "Maintain current clearance while monitoring separation."
    return (
        "Verified conflict-resolution plan for %s. Preference: %s. "
        "Instruction: %s Safety basis: the proposed commands were checked by the local forward verifier "
        "before this explanation was generated."
    ) % (conflict_text, preference, instruction_text)


class Handler(BaseHTTPRequestHandler):
    server_version = "ATCMockLLM/1.0"

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {}
        text = build_response_text(payload)
        body = {
            "id": "mock-atc-response",
            "object": "chat.completion",
            "model": payload.get("model", "mock-atc-explainer"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }
        data = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("Mock ATC LLM server listening on http://%s:%d/v1/chat/completions" % (args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
