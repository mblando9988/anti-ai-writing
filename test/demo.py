#!/usr/bin/env python3
# Demo: show HOW the detector decides. For each sentence it prints the verdict,
# the score, and the nearest AI and human example it matched against.
import json, os, subprocess, tempfile, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB  = os.path.join(ROOT, "bin", "embed_one")
HOOK = os.path.join(ROOT, "bin", "anti_ai_sem")
corpus = {r["text"]: r for r in json.load(open(os.path.join(ROOT, "corpus", "anti_ai_corpus.json")))}
ref = json.load(open(os.path.join(ROOT, "corpus", "corpus_emb.json")))

def unit(v):
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]
def cos(a, b): return sum(x*y for x, y in zip(a, b))

def vec(t): return json.loads(subprocess.run([EMB], input=t.encode(), capture_output=True, timeout=60).stdout or "[]")
def verdict(t):
    with tempfile.TemporaryDirectory() as td:
        tp = os.path.join(td, "t.jsonl")
        open(tp, "w").write(json.dumps({"type":"assistant","message":{"role":"assistant","content":t}})+"\n")
        p = subprocess.run([HOOK], input=json.dumps({"transcript_path":tp}).encode(), capture_output=True, timeout=60)
        return p.returncode, p.stderr.decode().strip()

def nearest(qv, label):
    q = unit(qv)
    best = sorted(((cos(q, unit(e["v"])), e["text"]) for e in ref if e["label"] == label), reverse=True)
    return best[0] if best else (0, "")

def show(t):
    rc, why = verdict(t)
    print("\n" + "─"*72)
    print(f'INPUT : "{t}"')
    print(f'VERDICT: {"⛔ FLAGGED AI (exit 2)" if rc==2 else "✅ reads human (exit 0)"}')
    print(f'        {why.split("): ",1)[-1] if "):" in why else why.split("allow: ")[-1]}')
    qv = vec(t)
    if len(qv) > 10:
        ca, ta = nearest(qv, "ai"); ch, th = nearest(qv, "human")
        print(f'  nearest AI    (cos {ca:.2f}): "{ta[:60]}"')
        print(f'  nearest HUMAN (cos {ch:.2f}): "{th[:60]}"')

SHOWCASE = [
 "Honestly, that's a fantastic question — let me delve into the nuances to truly empower you.",
 "I'm thrilled to announce our groundbreaking platform that unlocks your full potential.",
 "As an AI, I don't have personal experiences, but I can only imagine how exciting that felt.",
 "You're absolutely right, and your instinct here is brilliant and exactly on target.",
 "The bug is a null pointer on line 88. Add a guard before the deref and it goes away.",
 "Good question, but the data doesn't support that. The latency is the disk, not the query.",
 "No. That config flag was removed in version three.",
]
if __name__ == "__main__":
    print("anti-ai-writing demo — what it catches and why")
    for t in SHOWCASE: show(t)
