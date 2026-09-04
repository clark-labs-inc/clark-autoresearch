#!/usr/bin/env python3
"""OpenRouter proposer for the clark-autoresearch `optimize` loop.

Reads the proposer JSON contract on stdin

    {"parent": {"prompt": "..."},
     "reflective_dataset": {"prompt": [{"input": {"question","expected"},
                                          "output": "<model ans>", "score": float,
                                          "trajectory": {...}}, ...]},
     "components": ["prompt"],
     "history": "<ledger dossier>"}

calls OpenRouter to propose an improved system prompt, and prints the candidate
JSON

    {"prompt": "..."}

on stdout. The optimizer treats the candidate as an opaque dict; this script
specializes it to a single "prompt" component for the math <answer>-tag task.

Usage:
    python3 or_proposer.py                  # live (needs OPENROUTER_API_KEY)
    python3 or_proposer.py --dry-run        # canned improved prompt, no key/network

The API key is read from $OPENROUTER_API_KEY, or from ~/.openrouter_api_key.
On any OpenRouter error the proposer falls back to the parent prompt so the
loop never crashes on a provider hiccup.
"""

import argparse
import json
import os
import re
import sys
import urllib.request

DEFAULT_MODEL = "qwen/qwen3.5-flash-02-23"
KEY_PATH = os.path.expanduser("~/.openrouter_api_key")
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

DRY_RUN_PROMPT = (
    "You are a math assistant. Read the user's question, compute the answer, "
    "and respond with ONLY the final answer wrapped in <answer> tags, for "
    "example <answer>42</answer>. Do not include any explanation, reasoning, "
    "or any other text."
)


def load_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    if os.path.isfile(KEY_PATH):
        with open(KEY_PATH) as f:
            v = f.read().strip()
            if v:
                return v
    return None


def chat(model, system, user, key, temperature=0.7, timeout=30.0):
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


def extract_prompt(text):
    """Pull a prompt out of the model reply, tolerating several shapes."""
    # {"prompt": "..."} anywhere
    for m in re.finditer(r'\{.*?"prompt"\s*:\s*"(.*?)"\s*.*?\}', text, flags=re.DOTALL):
        return m.group(1).encode().decode("unicode_escape")
    # ```json { "prompt": "..." } ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))["prompt"]
        except Exception:
            pass
    # bare {"prompt": "..."} parse
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "prompt" in obj:
            return obj["prompt"]
    except Exception:
        pass
    # last resort: treat the whole reply as the prompt text
    return text.strip()


def main():
    p = argparse.ArgumentParser(description="OpenRouter proposer for optimize loop")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dry-run", action="store_true",
                   help="return a canned improved prompt (no key/network)")
    args = p.parse_args()

    raw = sys.stdin.read()
    try:
        req = json.loads(raw) if raw.strip() else {}
    except Exception:
        req = {}
    parent_prompt = (req.get("parent") or {}).get("prompt", "")
    rdataset = req.get("reflective_dataset") or {}
    history = req.get("history") or ""

    failures = []
    for entries in rdataset.values():
        for e in entries:
            inp = e.get("input") if isinstance(e.get("input"), dict) else {}
            if e.get("score", 0.0) < 1.0:
                failures.append(
                    "Q: {q}\nExpected: {exp}\nAssistant output: {out}\nScore: {sc}".format(
                        q=inp.get("question", ""),
                        exp=inp.get("expected", ""),
                        out=e.get("output", ""),
                        sc=e.get("score", 0.0),
                    )
                )

    if args.dry_run:
        new_prompt = DRY_RUN_PROMPT
    else:
        key = load_key()
        if not key:
            print(json.dumps({"prompt": parent_prompt}))
            return
        sys_msg = (
            "You optimize a SYSTEM PROMPT for a math question-answering assistant. "
            "The assistant MUST output its final answer wrapped in <answer> tags "
            'like <answer>42</answer>, with nothing else. Below are the current '
            "system prompt and the training examples it got WRONG (with the "
            "assistant's actual output). Propose an improved system prompt that "
            'makes the assistant reliably wrap its answer in <answer> tags. Reply '
            'with ONLY a JSON object {"prompt": "..."}.'
        )
        user_msg = (
            f"Current system prompt:\n{parent_prompt}\n\n"
            "Failed examples:\n"
            + ("\n---\n".join(failures) if failures else "(none so far)")
            + (f"\n\nRecent reflection history:\n{history}" if history else "")
        )
        try:
            txt = chat(args.model, sys_msg, user_msg, key)
            new_prompt = extract_prompt(txt)
        except Exception as e:
            sys.stderr.write(f"[proposer] openrouter error: {e}; reusing parent\n")
            new_prompt = parent_prompt

    print(json.dumps({"prompt": new_prompt}))


if __name__ == "__main__":
    main()