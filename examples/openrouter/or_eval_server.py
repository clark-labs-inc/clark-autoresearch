#!/usr/bin/env python3
"""OpenRouter eval server for the clark-autoresearch `optimize` loop.

Implements the `ResearchAdapter` eval contract over HTTP: a POST (any path)
with JSON

    {"candidate": {"prompt": "..."}, "batch": [{"question": "...", "expected": "..."}, ...],
     "capture_traces": bool}

returns

    {"scores": [float, ...], "outputs": [str, ...], "trajectories": [...],
     "num_metric_calls": int}

Task: the candidate is a *system prompt* for a math Q&A assistant. The
assistant must wrap its final answer in <answer>N</answer> tags. A response that
contains <answer>N</answer> with N == expected scores 1.0, otherwise 0.0. This
is deliberately prompt-sensitive: a weak "be helpful" prompt yields bare numbers
(score 0), while a prompt that instructs <answer> tags yields tagged answers
(score 1) -- so the optimization loop has a real signal to climb.

Usage:
    python3 or_eval_server.py --port 8081            # live (needs OPENROUTER_API_KEY)
    python3 or_eval_server.py --port 8081 --dry-run  # simulated model, no key/network

The API key is read from $OPENROUTER_API_KEY, or from ~/.openrouter_api_key.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_MODEL = "qwen/qwen3.5-flash-02-23"
KEY_PATHS = [os.path.expanduser("~/.openrouter_api_key")]
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def load_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for p in KEY_PATHS:
        if os.path.isfile(p):
            with open(p) as f:
                v = f.read().strip()
                if v:
                    return v
    return None


def chat(model, system, user, key, temperature=0.0, timeout=30.0):
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
    ).encode()
    req = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def extract_answer(text):
    """Return the number inside the last <answer>...</answer>, or None."""
    m = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    inner = m[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", inner)
    return nums[-1] if nums else inner


def score_response(resp, expected):
    ans = extract_answer(resp)
    if ans is None:
        return 0.0
    try:
        return 1.0 if float(ans) == float(expected) else 0.0
    except (ValueError, TypeError):
        return 1.0 if str(ans).strip() == str(expected).strip() else 0.0


def simulate(prompt, expected):
    """Mimic a prompt-sensitive model: tag the answer only if instructed to."""
    if "<answer>" in prompt.lower():
        return f"<answer>{expected}</answer>"
    return f"{expected}"


_args = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % a))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode() or "{}")
        except Exception as e:
            self._send(400, {"error": f"bad json: {e}"})
            return
        prompt = (payload.get("candidate") or {}).get("prompt", "")
        batch = payload.get("batch") or []
        capture = bool(payload.get("capture_traces", False))
        key = load_key()
        scores, outputs, trajectories = [], [], []
        for ex in batch:
            question = ex.get("question", "")
            expected = str(ex.get("expected", ""))
            err = None
            if _args.dry_run:
                resp = simulate(prompt, expected)
            elif not key:
                err = "no OPENROUTER_API_KEY"
                resp = ""
            else:
                try:
                    resp = chat(_args.model, prompt, question, key)
                except Exception as e:  # never crash the loop on one bad call
                    err = f"openrouter error: {e}"
                    resp = ""
            s = 0.0 if err else score_response(resp, expected)
            scores.append(s)
            outputs.append(resp)
            trajectories.append(
                {"raw": resp, "scored": extract_answer(resp), "error": err}
                if capture
                else None
            )
        n = len(batch)
        sys.stderr.write(
            f"[eval] prompt={prompt[:70]!r} scores={scores} calls={n}\n"
        )
        self._send(200, {"scores": scores, "outputs": outputs,
                         "trajectories": trajectories, "num_metric_calls": n})

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    global _args
    p = argparse.ArgumentParser(description="OpenRouter eval server for optimize loop")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dry-run", action="store_true",
                   help="simulate the model locally (no key/network)")
    _args = p.parse_args()
    if not _args.dry_run and not load_key():
        sys.stderr.write(
            "WARNING: no OPENROUTER_API_KEY (live mode will score 0 on all "
            "examples). Set $OPENROUTER_API_KEY or write ~/.openrouter_api_key.\n"
        )
    srv = ThreadingHTTPServer(("127.0.0.1", _args.port), Handler)
    sys.stderr.write(
        f"eval server on http://127.0.0.1:{_args.port} "
        f"(model={_args.model}, dry_run={_args.dry_run})\n"
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()