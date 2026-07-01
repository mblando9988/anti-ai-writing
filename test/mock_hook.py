#!/usr/bin/env python3
# Pure-python mock of bin/anti_ai_sem's I/O contract, so test_contract.py can
# exercise its own fixtures on machines that can't build the real binary.
# The verdict rule is deliberately trivial ("tapestry" => AI); everything else
# (guards, transcript parsing, SR_MARGIN, word floor, stderr format) must stay
# in lock-step with src/anti_ai_sem.swift.
import json, os, sys

def allow(why=""):
    if why:
        sys.stderr.write(f"[anti-ai-sem] allow: {why}\n")
    sys.exit(0)

def block(why):
    sys.stderr.write(f"BLOCKED by anti-ai-sem (reads as AI-generated): {why}\n")
    sys.exit(2)

raw = sys.stdin.buffer.read()
if not raw:
    allow("empty stdin")
try:
    obj = json.loads(raw)
except Exception:
    allow("invalid stdin")
if not isinstance(obj, dict):
    allow("invalid stdin")
if obj.get("stop_hook_active") is True:
    allow("stop_hook_active")

def last_assistant_text(path):
    try:
        content = open(path, encoding="utf-8").read()
    except Exception:
        return ""
    last = ""
    for line in content.split("\n"):
        try:
            o = json.loads(line)
        except Exception:
            continue
        if not isinstance(o, dict):
            continue
        m = o.get("message") if isinstance(o.get("message"), dict) else {}
        if o.get("type") != "assistant" and m.get("role") != "assistant":
            continue
        c = m.get("content")
        if isinstance(c, str) and c:
            last = c
        elif isinstance(c, list):
            t = " ".join(b["text"] for b in c
                         if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str))
            if t:
                last = t
    return last

text = ""
tp = obj.get("transcript_path")
if isinstance(tp, str):
    text = last_assistant_text(tp)
if not text and isinstance(obj.get("_text"), str):
    text = obj["_text"]
text = text.strip()
if not text:
    allow("empty turn")
if len(text.split()) < 6:
    allow("too short to judge style")

try:
    MARGIN = float(os.environ.get("SR_MARGIN", ""))
except ValueError:
    MARGIN = 0.02
margin = 0.5 if "tapestry" in text.lower() else -0.5
score = margin
if score > MARGIN:
    block(f"score={score:.2f} (margin={margin:.2f}, nudge=+0.00 —); evidence: mock")
allow(f"score={score:.2f} (margin={margin:.2f} nudge=+0.00) — reads human")
