#!/usr/bin/env python3
# Add an example, then rebuild. Usage:
#   python3 add.py ai    "the AI-sounding sentence"
#   python3 add.py human "a normal human sentence"
#   python3 add.py --no-rebuild ai "..."   # append only, rebuild embeddings later
import json, os, sys, subprocess
here = os.path.dirname(os.path.abspath(__file__))
cf = os.path.join(here, "corpus", "anti_ai_corpus.json")
ef = os.path.join(here, "corpus", "corpus_emb.json")

# flags are only honored BEFORE the label — a flag inside the example text must be
# refused, never silently stripped into the stored sentence
args = sys.argv[1:]
rebuild = True
while args and args[0] == "--no-rebuild":
    rebuild = False; args = args[1:]
if args and args[0].startswith("--"):
    print(f"unknown flag {args[0]!r}"); sys.exit(1)
if len(args) < 2 or args[0] not in ("ai", "human"):
    print('usage: python3 add.py [--no-rebuild] ai|human "your sentence"'); sys.exit(1)
label, text = args[0], " ".join(args[1:]).strip()
if any(a.startswith("--") for a in args[1:]):
    print("flags go before the label; refusing to store a flag as example text."); sys.exit(1)
if len(text.split()) < 6:
    print("too short: the hook ignores anything under 6 words, so this example could never matter."); sys.exit(1)

def norm(s): return " ".join(s.casefold().split())
data = json.load(open(cf))
try:
    embedded = {norm(e["text"]) for e in json.load(open(ef))}
except Exception:
    embedded = set()

if any(norm(r["text"]) == norm(text) for r in data):
    if norm(text) in embedded or not rebuild:
        print("already in the list, skipping."); sys.exit(0)
    print("already in the list but not embedded yet — rebuilding so it counts...")
else:
    data.append({"id": len(data), "label": label, "source": "mine", "text": text})
    json.dump(data, open(cf, "w"), indent=1, ensure_ascii=False)
    print(f'added as "{label}".')
    if not rebuild:
        print("skipped rebuild — the example is saved but has no embedding yet; run:")
        print("  cd corpus && swift ../src/embed_corpus.swift"); sys.exit(0)
    print("rebuilding (takes a minute)...")

try:
    subprocess.run(["swift", os.path.join(here, "src", "embed_corpus.swift")],
                   cwd=os.path.join(here, "corpus"), check=True)
except (FileNotFoundError, subprocess.CalledProcessError) as e:
    print(f"rebuild failed ({e}).")
    print("The example IS saved in the corpus, but it has no embedding yet, so the")
    print("hook won't use it as a reference until you rebuild (macOS):")
    print("  cd corpus && swift ../src/embed_corpus.swift")
    sys.exit(1)
print("done. it now knows that example.")
