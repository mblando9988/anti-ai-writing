#!/usr/bin/env python3
# Data quality gate for the corpus. Runs anywhere python3 runs.
#
# Hard failures (exit 1): broken schema, duplicate texts, id collisions,
# embeddings that point at texts no longer in the corpus.
# Warnings (exit 0): examples pending embedding, examples too short for the
# hook to ever judge, label imbalance, near-duplicate embeddings.
import json, math, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus", "anti_ai_corpus.json")
EMB = os.path.join(ROOT, "corpus", "corpus_emb.json")


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def main():
    corpus = json.load(open(CORPUS))
    errors, warnings = [], []

    # schema
    for i, x in enumerate(corpus):
        if not isinstance(x, dict) or set(x) < {"label", "text", "id"}:
            errors.append(f"item {i}: missing label/text/id fields")
            continue
        if x["label"] not in ("ai", "human"):
            errors.append(f"id {x['id']}: bad label {x['label']!r}")
        if not isinstance(x["text"], str) or not x["text"].strip():
            errors.append(f"id {x['id']}: empty text")
        if not isinstance(x["id"], int):
            errors.append(f"item {i}: non-integer id {x['id']!r}")

    # id collisions
    ids = [x["id"] for x in corpus]
    for d in sorted({i for i in ids if ids.count(i) > 1}):
        errors.append(f"id {d} used more than once")

    # duplicate texts (whitespace/case-insensitive)
    seen = {}
    for x in corpus:
        k = norm(x["text"])
        if k in seen:
            errors.append(f"duplicate text: id {seen[k]} and id {x['id']}: {x['text'][:60]!r}")
        seen[k] = x["id"]

    # too short for the hook's 6-word input gate
    for x in corpus:
        if len([w for w in re.split(r"[ \n]", x["text"]) if w]) < 6:
            warnings.append(f"id {x['id']}: under 6 words — the hook never judges "
                            f"text this short, so it only ever acts as a neighbor: "
                            f"{x['text'][:60]!r}")

    # balance
    n_ai = sum(1 for x in corpus if x["label"] == "ai")
    n_hu = len(corpus) - n_ai
    if n_hu and not (0.5 <= n_ai / n_hu <= 2.0):
        warnings.append(f"label imbalance: {n_ai} ai vs {n_hu} human")

    # embedding sync
    emb = json.load(open(EMB)) if os.path.exists(EMB) else []
    corpus_texts = {x["text"] for x in corpus}
    emb_texts = {x["text"] for x in emb}
    pending = corpus_texts - emb_texts
    orphans = emb_texts - corpus_texts
    if pending:
        warnings.append(f"{len(pending)} corpus examples have no embedding yet — "
                        f"run `swift src/embed_corpus.swift` in corpus/ on macOS")
    for t in sorted(orphans)[:5]:
        errors.append(f"embedding for text no longer in corpus: {t[:60]!r}")
    if len(orphans) > 5:
        errors.append(f"...and {len(orphans) - 5} more orphaned embeddings")

    # near-duplicate embeddings (informational; needs the vectors)
    if emb:
        try:
            import numpy as np
            v = np.asarray([x["v"] for x in emb], dtype=np.float64)
            n = v / np.linalg.norm(v, axis=1, keepdims=True)
            s = n @ n.T
            np.fill_diagonal(s, 0)
            pairs = {(min(i, j), max(i, j)) for i, j in zip(*np.where(s > 0.995))}
            for i, j in sorted(pairs):
                warnings.append(f"near-duplicate embeddings (cos={s[i][j]:.4f}): "
                                f"{emb[i]['text'][:50]!r} / {emb[j]['text'][:50]!r}")
        except ImportError:
            pass  # numpy is optional; skip the expensive check without it

    print(f"corpus: {len(corpus)} examples ({n_ai} ai / {n_hu} human), "
          f"{len(emb)} embedded, {len(pending)} pending")
    for w in warnings:
        print(f"  warn: {w}")
    for e in errors:
        print(f"  FAIL: {e}")
    print("RESULT:", "GREEN" if not errors else "BLOCKED")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
