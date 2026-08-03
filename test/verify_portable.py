#!/usr/bin/env python3
# The portable twin of verify_sem.py: runs EVERY corpus example through the
# real portable hook (src/anti_ai_sem.py) as a subprocess — actual stdin
# contract, actual exit codes — on any OS with python3. No swiftc, no Apple
# frameworks, no build step, no re-embed step.
#
# This is the "does it actually work, end to end, on all of them" test:
# each example is written into a transcript file, fed to the hook the same
# way the runtime feeds it, and judged by its exit code. The hook skips
# exact-text self matches, so every example is scored against the rest of
# the corpus, never against itself.
import json, os, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK = [sys.executable, os.path.join(ROOT, "src", "anti_ai_sem.py")]
CORPUS = json.load(open(os.path.join(ROOT, "corpus", "anti_ai_corpus.json")))
GATE_P, GATE_R = 0.99, 0.98


def run(text=None, raw=None):
    if raw is not None:
        return subprocess.run(HOOK, input=raw, capture_output=True, timeout=60).returncode
    with tempfile.TemporaryDirectory() as td:
        tp = os.path.join(td, "t.jsonl")
        open(tp, "w").write(json.dumps(
            {"type": "assistant", "message": {"role": "assistant", "content": text}}) + "\n")
        return subprocess.run(HOOK, input=json.dumps({"transcript_path": tp}).encode(),
                              capture_output=True, timeout=60).returncode


def main():
    corpus = CORPUS
    if "--sample" in sys.argv:
        # deterministic every-Nth thinning for slow environments (no numpy)
        want = int(sys.argv[sys.argv.index("--sample") + 1])
        step = max(1, len(corpus) // want)
        corpus = corpus[::step]
        print(f"sampling every {step}th item: {len(corpus)} of {len(CORPUS)}")
    t0 = time.time()
    TP = FP = FN = TN = 0
    fp, fn = [], []
    for i, c in enumerate(corpus, 1):
        blocked = run(text=c["text"]) == 2
        if c["label"] == "ai":
            if blocked: TP += 1
            else: FN += 1; fn.append(c)
        else:
            if blocked: FP += 1; fp.append(c)
            else: TN += 1
        if i % 50 == 0:
            print(f"  ...{i}/{len(CORPUS)} ({time.time()-t0:.0f}s)", flush=True)
    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / (TP + FN) if TP + FN else 0
    print(f"PORTABLE HOOK over {len(corpus)} items in {time.time()-t0:.0f}s")
    print(f"TP={TP} FP={FP} FN={FN} TN={TN}  precision={prec:.3f} recall={rec:.3f}  "
          f"(gate >={GATE_P} / >={GATE_R})")

    checks = [
        ("'delve' allows (too short)", run(text="delve") != 2),
        ("'ensure the cache is enabled' allows", run(text="ensure the cache is enabled") != 2),
        ("empty stdin fail-open", run(raw=b"") == 0),
        ("garbage stdin fail-open", run(raw=b"\xff not json {{{") == 0),
        ("stop_hook_active loop guard", run(raw=json.dumps(
            {"_text": "x " * 20, "stop_hook_active": True}).encode()) == 0),
    ]
    fails = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")

    print("-- false positives (human blocked) --")
    for c in fp[:10]:
        print(f"  FP {c['id']} [{c['source']}] {c['text'][:75]}")
    print("-- false negatives (ai allowed) --")
    for c in fn[:10]:
        print(f"  FN {c['id']} [{c['source']}] {c['text'][:75]}")

    green = prec >= GATE_P and rec >= GATE_R and not fails
    print("\nRESULT:", "GREEN" if green else "BLOCKED")
    sys.exit(0 if green else 1)


if __name__ == "__main__":
    main()
