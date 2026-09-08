#!/usr/bin/env python3
"""Tinker proxy — OpenAI-compatible HTTP shim that fronts Tinker's sampling
API. Lets api/provider (and any OpenAI SDK) talk to Source fine-tunes
without knowing they're hosted on Tinker.

Multi-sampler: routes by request.model to the configured sampler.

Config via env vars:
  TINKER_API_KEY        required — Tinker API key
  PORT                  optional, default 11435
  HOST                  optional, default 0.0.0.0
  SAMPLERS              required — JSON map:
    {
      "source-medium": {
        "sampler": "tinker://…/sampler_weights/source-medium-v6-2-medium",
        "base":    "Qwen/Qwen3.6-35B-A3B"
      },
      "source-vision": {
        "sampler": "tinker://…/sampler_weights/source-vision-v0-1-vision",
        "base":    "Qwen/Qwen3-VL-30B-A3B-Instruct"
      }
    }
  DEFAULT_MAX_TOKENS    optional, default 512
  DEFAULT_TEMPERATURE   optional, default 0.3
  DEFAULT_TOP_P         optional, default 0.92
  NO_THINK              optional, default "true" — prefill </think> so model
                        skips its thinking block (matches v6.2 training)
  INTERNAL_SECRET       required for /v1/* — shared bearer secret. Only
                        api/provider knows this. Generate with
                        `openssl rand -hex 32` and store the same value
                        in construct_upstream_providers.api_key_encrypted
                        for id='tinker'.
  ALLOWED_ORIGINS       optional — comma-separated list of CORS origins
                        allowed on preflight. Default empty = no browser-
                        origin requests (only server-to-server from
                        api/provider). Add e.g. "https://lisaos.dev"
                        only if you intentionally want browser direct.
  TRUST_PROXY_HEADERS   optional, default "true" — read X-Forwarded-For
                        for client IP logging. Caprover sets these.

API:
  GET  /health                                  — unauthed (Caprover probe)
  GET  /v1/models                               — auth required
  POST /v1/chat/completions                     — auth + X-Construct-User-Id

Required request headers on /v1/* (besides Authorization):
  X-Construct-User-Id   the end-user's id (forwarded by api/provider from
                        its gateway-injected X-Auth-User-ID). Logged per
                        request for per-user audit. Returns 400 if missing.
"""
import hmac
import json
import os
import re
import sys
import threading
import time
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import tinker
from tinker import types
from tinker.types import EncodedTextChunk, ModelInput


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _load_samplers() -> dict:
    raw = os.environ.get("SAMPLERS", "").strip()
    if not raw:
        sys.exit("SAMPLERS env var required (JSON map: name → {sampler, base})")
    try:
        cfg = json.loads(raw)
    except Exception as e:
        sys.exit(f"SAMPLERS env var is not valid JSON: {e}")
    if not isinstance(cfg, dict) or not cfg:
        sys.exit("SAMPLERS must be a non-empty JSON object")
    for name, entry in cfg.items():
        if not isinstance(entry, dict) or "sampler" not in entry or "base" not in entry:
            sys.exit(f"SAMPLERS['{name}'] must have keys: sampler, base")
    return cfg


def _load_allowed_origins() -> set:
    raw = os.environ.get("ALLOWED_ORIGINS", "").strip()
    if not raw:
        return set()
    return {o.strip() for o in raw.split(",") if o.strip()}


# ─── config ───────────────────────────────────────────────────────────────
CONFIG = {
    "host": os.environ.get("HOST", "0.0.0.0"),
    "port": int(os.environ.get("PORT", "11435")),
    "samplers": _load_samplers(),
    "default_max_tokens": int(os.environ.get("DEFAULT_MAX_TOKENS", "512")),
    "default_temperature": float(os.environ.get("DEFAULT_TEMPERATURE", "0.3")),
    "default_top_p": float(os.environ.get("DEFAULT_TOP_P", "0.92")),
    "no_think": _bool("NO_THINK", True),
    "internal_secret": os.environ.get("INTERNAL_SECRET", "").strip(),
    "allowed_origins": _load_allowed_origins(),
    "trust_proxy_headers": _bool("TRUST_PROXY_HEADERS", True),
}

if "TINKER_API_KEY" not in os.environ:
    sys.exit("TINKER_API_KEY env var required")

if not CONFIG["internal_secret"]:
    print("⚠️  INTERNAL_SECRET not set — /v1/* is OPEN. Safe only inside a private network.", file=sys.stderr, flush=True)


# ─── lazy sampling clients ────────────────────────────────────────────────
# One sampling client + tokenizer per configured sampler. Built on first
# use; cached for the process lifetime. Tinker's SDK is thread-safe for
# .sample() calls so a single client serves concurrent requests.
_state = {"clients": {}, "lock": threading.Lock(), "service": None}


def _service():
    if _state["service"] is None:
        _state["service"] = tinker.ServiceClient()
    return _state["service"]


def get_client(model_name: str):
    """Returns (sampling_client, tokenizer) for the given model name.
    Raises KeyError if model_name not in SAMPLERS."""
    if model_name not in CONFIG["samplers"]:
        raise KeyError(model_name)
    with _state["lock"]:
        cached = _state["clients"].get(model_name)
        if cached:
            return cached
        entry = CONFIG["samplers"][model_name]
        sc = _service().create_sampling_client(
            base_model=entry["base"], model_path=entry["sampler"],
        )
        tok = sc.get_tokenizer()
        _state["clients"][model_name] = (sc, tok)
        return sc, tok


# ─── chat helpers ─────────────────────────────────────────────────────────
def render_prompt(tok, messages, no_think: bool):
    """Apply chat template; if no_think, prefill an empty </think> block so
    the model starts past the open thinking block the template emits."""
    base = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if no_think:
        if base.endswith("<think>\n"):
            base = base + "\n</think>\n\n"
        elif "<think>" in base.split("assistant\n")[-1]:
            base = base + "<think>\n\n</think>\n\n"
    return tok(base, return_tensors=None)["input_ids"]


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_TRAIL_IMEND = re.compile(r"<\|im_end\|>\s*$")


def strip_think_artifacts(text: str) -> str:
    return _TRAIL_IMEND.sub("", _THINK_BLOCK.sub("", text)).strip()


def chat_completion(body: dict) -> dict:
    """Translate OpenAI chat → Tinker sample → OpenAI response."""
    model = body.get("model")
    if not model:
        return {"error": {"message": "field 'model' required", "type": "invalid_request_error"}, "_status": 400}
    try:
        sc, tok = get_client(model)
    except KeyError:
        return {
            "error": {
                "message": f"unknown model: {model}. Configured: {list(CONFIG['samplers'].keys())}",
                "type": "invalid_request_error",
            },
            "_status": 404,
        }
    messages = body.get("messages") or []
    if not messages:
        return {"error": {"message": "messages required", "type": "invalid_request_error"}, "_status": 400}

    # per-request no_think override (Qwen-style /think or /no_think trailing
    # the user message); falls back to global default
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    no_think = CONFIG["no_think"]
    if last_user and isinstance(last_user.get("content"), str):
        c = last_user["content"].rstrip()
        if c.endswith("/think"):
            no_think = False
        elif c.endswith("/no_think"):
            no_think = True

    ids = render_prompt(tok, messages, no_think=no_think)
    req = ModelInput(chunks=[EncodedTextChunk(tokens=ids)])
    params = types.SamplingParams(
        max_tokens=int(body.get("max_tokens", CONFIG["default_max_tokens"])),
        temperature=float(body.get("temperature", CONFIG["default_temperature"])),
        top_p=float(body.get("top_p", CONFIG["default_top_p"])),
    )

    fut = sc.sample(prompt=req, num_samples=1, sampling_params=params)
    resp = fut.result()
    seqs = getattr(resp, "samples", None) or getattr(resp, "sequences", None) or []
    tokens = list(getattr(seqs[0], "tokens", [])) if seqs else []
    raw_text = tok.decode(tokens) if tokens else ""
    content = strip_think_artifacts(raw_text)

    return {
        "id": "chatcmpl-tinker-" + str(int(time.time() * 1000)),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": len(ids), "completion_tokens": len(tokens), "total_tokens": len(ids) + len(tokens)},
    }


# ─── HTTP server ──────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "tinker-proxy/1.0"

    def log_message(self, fmt, *args):
        sys.stdout.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))
        sys.stdout.flush()

    def _allowed_origin(self) -> str:
        """Return the request's Origin if it's in the allowlist, else empty."""
        origin = self.headers.get("Origin", "")
        if origin and origin in CONFIG["allowed_origins"]:
            return origin
        return ""

    def _send(self, code: int, body: dict):
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        origin = self._allowed_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(data)

    def _client_ip(self) -> str:
        if CONFIG["trust_proxy_headers"]:
            xff = self.headers.get("X-Forwarded-For", "")
            if xff:
                return xff.split(",")[0].strip()
        return self.client_address[0] if self.client_address else "?"

    def _check_auth(self, path: str) -> tuple[bool, str]:
        """Return (ok, user_id). Sends a 401/400 response and returns
        (False, '') otherwise. /health is always open and skips user-id."""
        if path == "/health":
            return True, ""
        # Bearer secret (api/provider sends this from upstream.api_key_encrypted)
        if CONFIG["internal_secret"]:
            auth = self.headers.get("Authorization", "")
            if not auth.lower().startswith("bearer "):
                self._send(401, {"error": {"message": "missing bearer token", "type": "invalid_api_key"}})
                return False, ""
            token = auth[7:].strip()
            if not hmac.compare_digest(token, CONFIG["internal_secret"]):
                self._send(401, {"error": {"message": "invalid api key", "type": "invalid_api_key"}})
                return False, ""
        # User id (forwarded by api/provider from gateway X-Auth-User-ID).
        # Required on /v1/* — no per-user accounting without it.
        user_id = self.headers.get("X-Construct-User-Id", "").strip()
        if not user_id:
            self._send(400, {"error": {"message": "X-Construct-User-Id header required", "type": "invalid_request_error"}})
            return False, ""
        return True, user_id

    def do_GET(self):
        path = urlparse(self.path).path
        ok, user_id = self._check_auth(path)
        if not ok:
            return
        if path == "/health":
            self._send(200, {"status": "ok", "models": list(CONFIG["samplers"].keys())})
            return
        if path == "/v1/models":
            now = int(time.time())
            data = [
                {"id": name, "object": "model", "created": now, "owned_by": "construct"}
                for name in CONFIG["samplers"]
            ]
            self._send(200, {"object": "list", "data": data})
            return
        self._send(404, {"error": {"message": f"GET {path} not supported", "type": "invalid_request_error"}})

    def do_POST(self):
        path = urlparse(self.path).path
        ok, user_id = self._check_auth(path)
        if not ok:
            return
        if path != "/v1/chat/completions":
            self._send(404, {"error": {"message": f"POST {path} not supported", "type": "invalid_request_error"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw or b"{}")
        except Exception as e:
            self._send(400, {"error": {"message": f"invalid json: {e}", "type": "invalid_request_error"}})
            return
        t0 = time.time()
        wants_stream = bool(body.get("stream"))
        try:
            result = chat_completion(body)
            status = result.pop("_status", 200)
            if wants_stream and status == 200:
                self._send_sse(result)
            else:
                self._send(status, result)
            # Per-request audit line: user, model, tokens, latency.
            # Goes to stdout; Caprover's log tail picks it up.
            usage = result.get("usage", {}) if status == 200 else {}
            print(
                f'audit user={user_id} model={body.get("model","?")} '
                f'status={status} stream={wants_stream} '
                f'pt={usage.get("prompt_tokens",0)} '
                f'ct={usage.get("completion_tokens",0)} '
                f'ip={self._client_ip()} '
                f'latency_ms={int((time.time()-t0)*1000)}',
                flush=True,
            )
        except Exception as e:
            traceback.print_exc()
            if wants_stream:
                self._send_sse_error(str(e))
            else:
                self._send(500, {"error": {"message": str(e), "type": "internal_error"}})
            print(
                f'audit user={user_id} model={body.get("model","?")} '
                f'status=500 stream={wants_stream} err="{str(e)[:80]}" '
                f'ip={self._client_ip()} '
                f'latency_ms={int((time.time()-t0)*1000)}',
                flush=True,
            )

    def _send_sse(self, result: dict):
        """Emit one OpenAI-style chat.completion.chunk SSE event with the
        full content as a single delta, then [DONE]. We don't have token
        streaming from Tinker's non-streaming sample API, so this is
        "fake streaming" — but it satisfies OpenAI SDKs that require SSE
        format on stream=true requests."""
        content = result["choices"][0]["message"]["content"]
        chunk_id = result["id"].replace("chatcmpl-", "chatcmpl-chunk-")
        created = result["created"]
        model = result["model"]
        first_chunk = {
            "id": chunk_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
        }
        final_chunk = {
            "id": chunk_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")  # disable nginx buffering
        origin = self._allowed_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        for chunk in (first_chunk, final_chunk):
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send_sse_error(self, msg: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        err = {"error": {"message": msg, "type": "internal_error"}}
        self.wfile.write(f"data: {json.dumps(err)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_OPTIONS(self):
        # Only respond with CORS if the request's Origin is allowlisted.
        # Otherwise return 204 with no CORS headers → browser blocks.
        origin = self._allowed_origin()
        self.send_response(204)
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Construct-User-Id")
            self.send_header("Vary", "Origin")
        self.end_headers()


def main():
    print(f"# tinker-proxy listening on http://{CONFIG['host']}:{CONFIG['port']}", flush=True)
    print(f"#   models: {list(CONFIG['samplers'].keys())}", flush=True)
    print(f"#   no_think default: {CONFIG['no_think']}", flush=True)
    print(f"#   internal_secret: {'set' if CONFIG['internal_secret'] else 'OPEN (no auth)'}", flush=True)
    print(f"#   allowed_origins: {sorted(CONFIG['allowed_origins']) or 'none (server-to-server only)'}", flush=True)
    server = HTTPServer((CONFIG["host"], CONFIG["port"]), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
