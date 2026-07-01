#!/usr/bin/env python3
# Input-contract tests for the hook: transcript parsing, fail-open guards,
# the SR_MARGIN override, the 6-word floor, and the stderr format the demo
# scripts parse. This is the production input path (Claude Code stop hook),
# so it gets pinned check by check.
#
# Needs bin/anti_ai_sem (macOS). On any machine, `--mock` runs the same
# fixtures against test/mock_hook.py, which mirrors the contract — that
# validates the suite itself, not the binary.
import json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "--mock" in sys.argv:
    HOOK = [sys.executable, os.path.join(ROOT, "test", "mock_hook.py")]
    print("running against test/mock_hook.py (contract mirror, trivial verdict rule)")
else:
    b = os.environ.get("HOOK_BIN", os.path.join(ROOT, "bin", "anti_ai_sem"))
    if not os.path.exists(b):
        print("bin/anti_ai_sem not built (macOS: swiftc -O src/anti_ai_sem.swift -o bin/anti_ai_sem).")
        print("Run `python3 test/test_contract.py --mock` to exercise the suite without it.")
        sys.exit(3)
    HOOK = [b]

# Known-verdict texts: BLOCK_TEXT is the strongest positive case in test/cases,
# PASS_TEXT is quick_test's normal reply. Both contain/lack "tapestry" so the
# mock's trivial rule agrees with the real binary's verdict on them.
BLOCK_TEXT = "Great question! I'd be happy to walk you through the intricate tapestry of considerations here."
PASS_TEXT = "The bug is a null pointer on line 88. Add a guard before the deref and it goes away. Tests pass after that."

td = tempfile.mkdtemp()
def transcript(lines, name):
    p = os.path.join(td, name + ".jsonl")
    with open(p, "w") as f:
        for l in lines:
            f.write((l if isinstance(l, str) else json.dumps(l)) + "\n")
    return p

def a_msg(content): return {"type": "assistant", "message": {"role": "assistant", "content": content}}
def u_msg(content): return {"type": "user", "message": {"role": "user", "content": content}}

def run(stdin, env=None):
    e = dict(os.environ); e.pop("SR_MARGIN", None); e.update(env or {})
    data = stdin if isinstance(stdin, bytes) else json.dumps(stdin).encode()
    p = subprocess.run(HOOK, input=data, capture_output=True, timeout=60, env=e)
    return p.returncode, p.stderr.decode()

fails = []
def check(name, got, want):
    ok = got == want
    if not ok: fails.append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} (exit {got}, want {want})")

def check_stderr(name, msg, needle):
    ok = needle in msg
    if not ok: fails.append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} (stderr {'contains' if ok else 'MISSING'} {needle!r})")

print("-- fail-open guards --")
check("empty stdin", run(b"")[0], 0)
check("garbage bytes", run(b"\xff{{{ not json")[0], 0)
check("JSON but not an object", run(b"[1, 2, 3]")[0], 0)
check("object with no text source", run({"other": "keys"})[0], 0)
check("stop_hook_active loop guard", run({"stop_hook_active": True, "_text": BLOCK_TEXT})[0], 0)
check("missing transcript file", run({"transcript_path": os.path.join(td, "nope.jsonl")})[0], 0)
check("whitespace-only text", run({"_text": "   \n\t  "})[0], 0)

print("-- transcript parsing --")
check("single message, string content",
      run({"transcript_path": transcript([a_msg(BLOCK_TEXT)], "p1")})[0], 2)
check("array-of-blocks content",
      run({"transcript_path": transcript(
          [a_msg([{"type": "text", "text": BLOCK_TEXT}])], "p2")})[0], 2)
check("uses LAST assistant message (pass, then slop)",
      run({"transcript_path": transcript(
          [u_msg("hi"), a_msg(PASS_TEXT), u_msg("more?"), a_msg(BLOCK_TEXT)], "p3")})[0], 2)
check("uses LAST assistant message (slop, then pass)",
      run({"transcript_path": transcript(
          [a_msg(BLOCK_TEXT), u_msg("thanks"), a_msg(PASS_TEXT)], "p4")})[0], 0)
check("tool_use-only final message keeps previous text",
      run({"transcript_path": transcript(
          [a_msg(BLOCK_TEXT), a_msg([{"type": "tool_use", "name": "Bash", "input": {}}])], "p5")})[0], 2)
check("empty-string final content keeps previous text",
      run({"transcript_path": transcript([a_msg(BLOCK_TEXT), a_msg("")], "p6")})[0], 2)
check("malformed line mid-file is skipped",
      run({"transcript_path": transcript(
          [a_msg(PASS_TEXT), "{{{{not json", a_msg(BLOCK_TEXT)], "p7")})[0], 2)
check("user messages are ignored (slop in user turn)",
      run({"transcript_path": transcript([u_msg(BLOCK_TEXT), a_msg(PASS_TEXT)], "p8")})[0], 0)
check("_text fallback when no transcript_path", run({"_text": BLOCK_TEXT})[0], 2)

print("-- SR_MARGIN override and the 6-word floor --")
check("5 words: floor wins even at SR_MARGIN=-999",
      run({"_text": "Delve into the vibrant tapestry"}, {"SR_MARGIN": "-999"})[0], 0)
check("6 words: judged (SR_MARGIN=-999 blocks)",
      run({"_text": "Delve into the vibrant tapestry now"}, {"SR_MARGIN": "-999"})[0], 2)
check("6 words split by tabs/newlines count as words",
      run({"_text": "Delve\tinto\tthe\nvibrant\ttapestry\tnow"}, {"SR_MARGIN": "-999"})[0], 2)
check("SR_MARGIN=999 lets strong slop pass",
      run({"_text": BLOCK_TEXT}, {"SR_MARGIN": "999"})[0], 0)
check("SR_MARGIN=garbage falls back to default",
      run({"_text": BLOCK_TEXT}, {"SR_MARGIN": "abc"})[0], 2)

print("-- stderr format (demo scripts parse this) --")
rc, msg = run({"_text": BLOCK_TEXT})
check_stderr("block message prefix", msg, "BLOCKED by anti-ai-sem")
check_stderr("block message has score=", msg, "score=")
rc, msg = run({"_text": PASS_TEXT})
check_stderr("allow message marker", msg, "allow:")

print(f"\n{len(fails)} failing" if fails else "\nall contract checks pass")
sys.exit(1 if fails else 0)
