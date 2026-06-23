#!/usr/bin/env python3
# Quick sanity test: slop should block, a normal reply should pass.
import json, os, subprocess, tempfile, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK = os.path.join(ROOT, "bin", "anti_ai_sem")
def check(text):
    with tempfile.TemporaryDirectory() as td:
        tp = os.path.join(td, "t.jsonl")
        open(tp, "w").write(json.dumps({"type":"assistant","message":{"role":"assistant","content":text}}) + "\n")
        p = subprocess.run([HOOK], input=json.dumps({"transcript_path": tp}).encode(),
                            capture_output=True, timeout=60)
        return p.returncode, p.stderr.decode().strip()
slop = "Honestly, that's a fantastic question. Let me delve into the nuances and break this down to truly empower you and unlock your full potential."
normal = "The bug is a null pointer on line 88. Add a guard before the deref and it goes away. Tests pass after that."
sc, sm = check(slop); nc, nm = check(normal)
print("slop   ->", "BLOCK (good)" if sc == 2 else "passed (BAD)", "|", sm.split('):')[-1].strip()[:50])
print("normal ->", "passed (good)" if nc == 0 else "BLOCK (BAD)", "|", nm.split('allow:')[-1].strip()[:50])
ok = sc == 2 and nc == 0
print("\nRESULT:", "PASS - blocks slop, allows normal" if ok else "FAIL")
sys.exit(0 if ok else 1)
