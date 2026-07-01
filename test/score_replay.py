#!/usr/bin/env python3
# Offline replay of the hook's verdict over the frozen embeddings. Runs
# anywhere — no macOS, no binary, no deps. This is how scoring changes get
# measured on machines that can't build the hook: it ports the margin + nudge
# math exactly, and a parity gate asserts the ported regexes still appear
# verbatim in src/anti_ai_sem.swift, so the two can't drift silently.
import json, math, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = open(os.path.join(ROOT, "src", "anti_ai_sem.swift")).read()
emb = json.load(open(os.path.join(ROOT, "corpus", "corpus_emb.json")))

# these must be EXACTLY equal to a #"..."# raw string in src/anti_ai_sem.swift;
# update both files together or the gate fails. Equality (not substring) matters:
# an alternative appended to the swift pattern must trip the gate, not slip past it.
VALIDATION = r"your (theory|hypothesis|framing|instinct|intuition|perspective|premise) (is|sounds) (spot[- ]on|right|correct|valid|sound|compelling|fascinating|brilliant|insightful|astute|excellent|great|perceptive|sharp|exactly|precisely|completely|absolutely|genuinely|impressively|refreshingly|entirely)|you raise (a )?(great|good|valid|important) point|completely (valid|understandable)|\b(is|are|sounds|you'?re|that'?s) spot[- ]on\b|on the right track|you'?re (absolutely )?right"
DISAGREE = r"\bhowever\b|\bin fact\b|not (quite|the case|true|right|the|outside|inside)\b|i'?d push back|the (data|evidence) (does|doesn'?t|suggest|show)|the opposite|but no\b|but not\b|but the\b|i misread|absolutely not|absolutely no\b"
OPENER_ASSIST = r"^\W*let me (delineate|lay out|unpack|walk you through|break (this|it) down|take a moment)|^\W*let'?s (dive|unpack|break (this|it) down|take a step back)"
OPENER_WORDS = r"^\W*(honestly|frankly|absolutely|certainly|of course)\b"
OPENER_PRAISE_Q = r"^\W*(great|excellent|fantastic|brilliant|wonderful|amazing|perfect)\s+(question|point|observation)"

raw_strings = set(re.findall(r'#"(.*?)"#', SRC, re.S))
parity_fail = [name for name, pat in [
    ("validation", VALIDATION), ("disagree", DISAGREE),
    ("assistant-opener", OPENER_ASSIST), ("opener-words", OPENER_WORDS),
    ("opener-praise-question", OPENER_PRAISE_Q),
] if pat not in raw_strings]
if parity_fail:
    print("PARITY FAIL: these replay patterns no longer match src/anti_ai_sem.swift:", ", ".join(parity_fail))
    print("Update the swift source and this file together.")
    sys.exit(1)

def nudge(text):
    text = text.replace("’", "'").replace("‘", "'")  # hook normalizes curly quotes
    lower = text.lower()
    opening = text[:120].lower()
    opener_praise = bool(re.search(OPENER_PRAISE_Q, lower, re.I)) or bool(re.match(OPENER_WORDS, opening))
    n = 0.0
    if opener_praise: n += 0.05
    if re.search(OPENER_ASSIST, lower): n += 0.03
    if re.search(VALIDATION, lower): n += 0.03
    if re.search(DISAGREE, lower): n -= 0.06
    return n

def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

texts = [e["text"] for e in emb]
labels = [e["label"] for e in emb]
U = [unit(e["v"]) for e in emb]

def dot(a, b): return sum(x * y for x, y in zip(a, b))

MARGIN = 0.02
TP = FP = FN = TN = 0
errs = []
for i, t in enumerate(texts):
    ai, hu = [], []
    for j in range(len(texts)):
        if texts[j] == t:  # the hook skips exact-text self matches
            continue
        (ai if labels[j] == "ai" else hu).append(dot(U[i], U[j]))
    top = lambda a: sum(sorted(a, reverse=True)[:5]) / min(5, len(a)) if a else 0
    score = top(ai) - top(hu) + nudge(t)
    blocked = score > MARGIN
    if labels[i] == "ai":
        TP, FN = TP + blocked, FN + (not blocked)
        if not blocked: errs.append(("FN", score, t))
    else:
        FP, TN = FP + blocked, TN + (not blocked)
        if blocked: errs.append(("FP", score, t))

prec = TP / (TP + FP) if TP + FP else 0
rec = TP / (TP + FN) if TP + FN else 0
print(f"replay over {len(texts)} embedded items: TP={TP} FP={FP} FN={FN} TN={TN} precision={prec:.3f} recall={rec:.3f}")
for kind, score, t in errs:
    print(f"  {kind} score={score:+.3f} {t[:80]!r}")
ok = prec >= 0.95 and rec >= 0.90
print("RESULT:", "OK" if ok else "FAIL (need precision>=0.95 recall>=0.90)")
sys.exit(0 if ok else 1)
