#!/usr/bin/env python3
# Add an example, then rebuild. Usage:
#   python3 add.py ai   "the AI-sounding sentence"
#   python3 add.py human "a normal human sentence"
# Pass --no-rebuild to just append (e.g. batching several before one rebuild).
import json, os, re, shutil, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
cf = os.path.join(here, "corpus", "anti_ai_corpus.json")
args = [a for a in sys.argv[1:] if a != "--no-rebuild"]
rebuild = "--no-rebuild" not in sys.argv
if len(args) < 2 or args[0] not in ("ai", "human"):
    print('usage: python3 add.py [--no-rebuild] ai|human "your sentence"'); sys.exit(1)
label, text = args[0], " ".join(args[1:]).strip()

def norm(t): return re.sub(r"\s+", " ", t.strip().lower())
if len(text.split()) < 6:
    print("that's under 6 words — the hook never judges text that short, "
          "so this example would never fire. give it a fuller sentence.")
    sys.exit(1)
data = json.load(open(cf))
if any(norm(r["text"]) == norm(text) for r in data):
    print("already in the list, skipping."); sys.exit(0)
data.append({"label": label, "source": "mine", "text": text,
             "id": max(r["id"] for r in data) + 1})
json.dump(data, open(cf, "w"), indent=1, ensure_ascii=False)
n_ai = sum(1 for r in data if r["label"] == "ai")
print(f'added as "{label}". corpus is now {len(data)} ({n_ai} ai / {len(data)-n_ai} human).')
if not rebuild:
    print("skipped rebuild (--no-rebuild). run `swift src/embed_corpus.swift` in corpus/ when done.")
    sys.exit(0)
if not shutil.which("swift"):
    print("swift isn't available here, so the embedding wasn't rebuilt. the example is "
          "saved; run `swift src/embed_corpus.swift` in corpus/ on a Mac to teach it.")
    sys.exit(0)
print("rebuilding (takes a minute)...")
subprocess.run(["swift", os.path.join(here, "src", "embed_corpus.swift")],
               cwd=os.path.join(here, "corpus"), check=True)
print("done. it now knows that example.")
