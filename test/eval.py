#!/usr/bin/env python3
# Real eval: run the hook over every labeled case in test/cases/{positive,negative,edge}.
# Each case is a .txt (the assistant message) + .expected (exit code: 2=block, 0=pass).
# Reports accuracy and, crucially, the two error types:
#   false positive = a human/edge case that got BLOCKED (the over-blocking the devil flagged)
#   false negative = AI slop that PASSED
import json, os, subprocess, tempfile, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK = os.path.join(ROOT, "bin", "anti_ai_sem")
CASES = os.path.join(ROOT, "test", "cases")

def run(text):
    with tempfile.TemporaryDirectory() as td:
        tp = os.path.join(td, "t.jsonl")
        open(tp, "w").write(json.dumps(
            {"type": "assistant", "message": {"role": "assistant", "content": text}}) + "\n")
        p = subprocess.run([HOOK], input=json.dumps({"transcript_path": tp}).encode(),
                           capture_output=True, timeout=60)
        return p.returncode, (p.stderr.decode().strip() or p.stdout.decode().strip())

def main():
    cases = sorted(glob.glob(os.path.join(CASES, "*", "*.txt")))
    if not cases:
        print("no cases found under", CASES); sys.exit(1)
    total = correct = fp = fn = 0
    fails = []
    for txt in cases:
        group = os.path.basename(os.path.dirname(txt))
        name = group + "/" + os.path.basename(txt)[:-4]
        text = open(txt).read()
        exp = int(open(txt[:-4] + ".expected").read().strip())
        rc, msg = run(text)
        got_block = (rc == 2)
        exp_block = (exp == 2)
        total += 1
        if got_block == exp_block:
            correct += 1
        else:
            if got_block and not exp_block:  # blocked a human -> false positive
                fp += 1; kind = "FALSE POSITIVE (blocked human)"
            else:                            # passed slop -> false negative
                fn += 1; kind = "FALSE NEGATIVE (passed slop)"
            fails.append((name, kind, msg.split("):")[-1].strip()[:70]))
    print(f"cases={total}  correct={correct}  accuracy={correct/total:.1%}")
    print(f"false positives (human blocked) = {fp}")
    print(f"false negatives (slop passed)   = {fn}")
    if fails:
        print("\nfailures:")
        for n, k, m in fails:
            print(f"  {n:32s} {k}\n      {m}")
    sys.exit(0 if not fails else 1)

if __name__ == "__main__":
    main()
