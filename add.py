#!/usr/bin/env python3
# Add an example, then rebuild. Usage:
#   python3 add.py ai   "the AI-sounding sentence"
#   python3 add.py human "a normal human sentence"
import json, os, sys, subprocess
here = os.path.dirname(os.path.abspath(__file__))
cf = os.path.join(here, "corpus", "anti_ai_corpus.json")
if len(sys.argv) < 3 or sys.argv[1] not in ("ai", "human"):
    print('usage: python3 add.py ai|human "your sentence"'); sys.exit(1)
label, text = sys.argv[1], " ".join(sys.argv[2:]).strip()
data = json.load(open(cf))
if any(r["text"] == text for r in data):
    print("already in the list, skipping."); sys.exit(0)
data.append({"id": len(data), "label": label, "source": "mine", "text": text})
json.dump(data, open(cf, "w"), indent=1, ensure_ascii=False)
print(f'added as "{label}". rebuilding (takes a minute)...')
subprocess.run(["swift", os.path.join(here, "src", "embed_corpus.swift")],
               cwd=os.path.join(here, "corpus"), check=True)
print("done. it now knows that example.")
