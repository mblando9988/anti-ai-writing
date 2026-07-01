#!/usr/bin/env python3
# Dataset hygiene check. Runs anywhere (no macOS, no deps).
# Hard errors (exit 1): duplicate texts, bad labels, non-contiguous ids,
#   items under the 6-word floor the hook ignores, stale or mislabeled embeddings.
# Warnings: corpus items not yet embedded (run src/embed_corpus.swift on a Mac).
# With numpy installed it also reports near-duplicate and cross-label-confusable
# pairs among the embedded items.
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
corpus = json.load(open(os.path.join(ROOT, "corpus", "anti_ai_corpus.json")))
emb = json.load(open(os.path.join(ROOT, "corpus", "corpus_emb.json")))

errors, warnings = [], []

texts = [c["text"] for c in corpus]
if len(texts) != len(set(texts)):
    seen, dups = set(), set()
    for t in texts:
        (dups if t in seen else seen).add(t)
    errors.append(f"{len(dups)} duplicate texts, e.g. {sorted(dups)[0][:60]!r}")
bad = [c["id"] for c in corpus if c.get("label") not in ("ai", "human")]
if bad: errors.append(f"bad labels on ids {bad[:10]}")
if [c["id"] for c in corpus] != list(range(len(corpus))):
    errors.append("ids are not contiguous 0..N-1 (add.py assumes id=len(data))")
short = [c["id"] for c in corpus if len(c["text"].split()) < 6]
if short: errors.append(f"items under 6 words (hook auto-allows them, dead weight): ids {short[:10]}")
nosrc = [c["id"] for c in corpus if not c.get("source")]
if nosrc: errors.append(f"missing source on ids {nosrc[:10]}")

by_text = {c["text"]: c for c in corpus}
stale = [e["text"][:60] for e in emb if e["text"] not in by_text]
if stale: errors.append(f"{len(stale)} embeddings for texts no longer in the corpus, e.g. {stale[0]!r}")
mislabeled = [e["text"][:60] for e in emb if e["text"] in by_text and by_text[e["text"]]["label"] != e["label"]]
if mislabeled: errors.append(f"{len(mislabeled)} label mismatches corpus vs embeddings, e.g. {mislabeled[0]!r}")
embedded = {e["text"] for e in emb}
pending = [c for c in corpus if c["text"] not in embedded]
if pending:
    warnings.append(f"{len(pending)} corpus items have no embedding yet — they are eval-only until "
                    f"you run `swift src/embed_corpus.swift` (from corpus/, macOS)")

ai = sum(1 for c in corpus if c["label"] == "ai")
print(f"corpus: {len(corpus)} items (ai={ai}, human={len(corpus)-ai}); embedded: {len(emb)}")

try:
    import numpy as np
    idx = [i for i, e in enumerate(emb)]
    V = np.array([e["v"] for e in emb])
    V = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)
    S = V @ V.T
    lbl = np.array([1 if e["label"] == "ai" else 0 for e in emb])
    iu = np.triu_indices(len(emb), k=1)
    same = (lbl[iu[0]] == lbl[iu[1]])
    ndup = int(((S[iu] > 0.995) & same).sum())
    conf = int(((S[iu] > 0.97) & ~same).sum())
    print(f"embedded-pair report: {ndup} same-label near-dups (cos>0.995), {conf} cross-label confusables (cos>0.97)")
    if conf:
        warnings.append(f"{conf} cross-label pairs above 0.97 cosine — an ai ref that close to a human ref "
                        f"poisons top-5 neighborhoods; consider dropping or relabeling one side")
except ImportError:
    print("(numpy not installed, skipping similarity report)")

for w in warnings: print("WARN:", w)
for e in errors: print("ERROR:", e)
print("RESULT:", "FAIL" if errors else "OK")
sys.exit(1 if errors else 0)
