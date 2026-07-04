#!/usr/bin/env python3
# Cross-platform backtest: replays the hook's exact scoring math over the
# precomputed corpus embeddings (corpus/corpus_emb.json), leave-one-out.
# No macOS, no swiftc, no NaturalLanguage needed — runs anywhere python3 runs.
#
# What it replicates from src/anti_ai_sem.swift, faithfully:
#   * top-5 mean cosine similarity to AI refs minus top-5 mean to human refs
#     (the contrastive margin), excluding exact-text self matches
#   * the bounded positional nudge (opener-praise +0.05, validation +0.03,
#     disagreement -0.06), same regexes
#   * verdict: score > MARGIN (default 0.02, override with SR_MARGIN)
#
# What it can't replicate: embedding *new* text (NLEmbedding is Apple-only).
# For that, build the binary on macOS and run test/verify_sem.py.
#
# Usage:
#   python3 test/backtest.py            # metrics + sweeps + error listing
#   python3 test/backtest.py --json out.json
#   SR_MARGIN=0.03 python3 test/backtest.py
#
# Exits 0 when leave-one-out precision >= 0.99 and recall >= 0.95, else 1.
import json, math, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(ROOT, "corpus", "corpus_emb.json")
CORPUS = os.path.join(ROOT, "corpus", "anti_ai_corpus.json")
K = 5
GATE_P, GATE_R = 0.99, 0.95

try:
    import numpy as np
except ImportError:
    np = None


def word_count(text):
    # Swift: text.split(whereSeparator: { $0 == " " || $0 == "\n" }).count
    return len([w for w in re.split(r"[ \n]", text) if w])


def nudge(text):
    """The positional nudge from anti_ai_sem.swift, same regexes, same weights."""
    lower = text.lower()
    opening = text[:120].lower()
    opener_praise = bool(re.match(
        r"^\W*(great|excellent|fantastic|brilliant|wonderful|amazing|perfect)"
        r"\s+(question|point|observation)", text, re.I)) or bool(re.match(
        r"^\W*(honestly|frankly|absolutely|certainly|of course)\b", opening))
    validation = bool(re.search(
        r"your (theory|hypothesis|framing|instinct|intuition|perspective|premise) (is|sounds)"
        r"|you raise (a )?(great|good|valid|important) point"
        r"|completely (valid|understandable)|spot on|on the right track"
        r"|you'?re (absolutely )?right", lower))
    # "you're not wrong" is validation wearing a negation costume — it praises,
    # it doesn't push back. Count it as validation and keep it away from the
    # disagreement redemption below.
    not_wrong = bool(re.search(r"you'?re not wrong|you are not wrong", lower))
    if not_wrong:
        validation = True
    sans_not_wrong = re.sub(r"you'?re not wrong|you are not wrong", "", lower)
    disagree = bool(re.search(
        r"\bhowever\b|\bin fact\b|not (quite|the case|true|wrong|right|the|outside|inside)\b"
        r"|i'?d push back|the (data|evidence) (does|doesn'?t|suggest|show)"
        r"|the opposite|but no\b|but not\b|but the\b|i misread"
        r"|absolutely not|absolutely no\b", sans_not_wrong))
    n, why = 0.0, []
    if opener_praise:
        n += 0.05; why.append("opener-praise")
    if validation:
        n += 0.03; why.append("validation")
    if disagree:
        n -= 0.06; why.append("disagreement(redeems)")
    return n, why


def cosine_matrix(vs):
    if np is not None:
        v = np.asarray(vs, dtype=np.float64)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        n = v / norms
        return (n @ n.T).tolist()
    # stdlib fallback: slower (~a minute on 500 items) but zero dependencies
    normed = []
    for v in vs:
        nrm = math.sqrt(sum(x * x for x in v)) or 1.0
        normed.append([x / nrm for x in v])
    size = len(normed)
    sims = [[0.0] * size for _ in range(size)]
    for i in range(size):
        vi = normed[i]
        sims[i][i] = 1.0
        for j in range(i + 1, size):
            s = sum(a * b for a, b in zip(vi, normed[j]))
            sims[i][j] = sims[j][i] = s
    return sims


def margins_at_k(items, sims, k):
    """Leave-one-out contrastive margin per item, excluding exact-text matches
    (the hook skips refs whose text equals the query text)."""
    out = []
    for i, it in enumerate(items):
        ai, hu = [], []
        for j, other in enumerate(items):
            if other["text"] == it["text"]:
                continue
            (ai if other["label"] == "ai" else hu).append(sims[i][j])
        ai.sort(reverse=True); hu.sort(reverse=True)
        top_ai = ai[:k]; top_hu = hu[:k]
        m_ai = sum(top_ai) / len(top_ai) if top_ai else 0.0
        m_hu = sum(top_hu) / len(top_hu) if top_hu else 0.0
        out.append(m_ai - m_hu)
    return out


def metrics(items, scores, thr):
    tp = fp = fn = tn = 0
    for it, s in zip(items, scores):
        flagged = s > thr and word_count(it["text"]) >= 6
        if it["label"] == "ai":
            tp, fn = (tp + 1, fn) if flagged else (tp, fn + 1)
        else:
            fp, tn = (fp + 1, tn) if flagged else (fp, tn + 1)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=p, recall=r, f1=f1)


def main():
    json_out = None
    if "--json" in sys.argv:
        json_out = sys.argv[sys.argv.index("--json") + 1]

    items = json.load(open(EMB))
    thr = float(os.environ.get("SR_MARGIN", "0.02"))
    n_ai = sum(1 for x in items if x["label"] == "ai")
    print(f"backtest over {len(items)} embedded examples "
          f"({n_ai} ai / {len(items) - n_ai} human), k={K}, threshold={thr}")

    # corpus items that don't have embeddings yet (added but not re-embedded)
    try:
        corpus = json.load(open(CORPUS))
        embedded = {x["text"] for x in items}
        pending = [c for c in corpus if c["text"] not in embedded]
        if pending:
            print(f"note: {len(pending)} corpus examples have no embedding yet — "
                  f"run `swift src/embed_corpus.swift` in corpus/ on macOS to include them")
    except FileNotFoundError:
        pending = []

    sims = cosine_matrix([x["v"] for x in items])
    margins = margins_at_k(items, sims, K)
    nudges = [nudge(x["text"])[0] for x in items]
    scores = [m + nd for m, nd in zip(margins, nudges)]

    base = metrics(items, scores, thr)
    print(f"\nleave-one-out @ threshold {thr}: "
          f"TP={base['TP']} FP={base['FP']} FN={base['FN']} TN={base['TN']}  "
          f"precision={base['precision']:.3f} recall={base['recall']:.3f} f1={base['f1']:.3f}")
    margin_only = metrics(items, margins, thr)
    print(f"embedding margin alone (no nudge):  "
          f"precision={margin_only['precision']:.3f} recall={margin_only['recall']:.3f} "
          f"(the nudge buys back {base['TP'] - margin_only['TP']} true positives)")

    print("\nthreshold sweep (with nudge):")
    sweep = {}
    for t in (0.00, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04):
        m = metrics(items, scores, t)
        sweep[t] = m
        mark = " <- current" if abs(t - thr) < 1e-9 else ""
        print(f"  thr={t:.3f}  P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f}  FP={m['FP']} FN={m['FN']}{mark}")

    print("\nk sweep (at current threshold):")
    ksweep = {}
    for k in (3, 5, 7, 9):
        sc = [m + nd for m, nd in zip(margins_at_k(items, sims, k), nudges)]
        m = metrics(items, sc, thr)
        ksweep[k] = m
        mark = " <- current" if k == K else ""
        print(f"  k={k}  P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f}  FP={m['FP']} FN={m['FN']}{mark}")

    fns = [(s, it) for s, it in zip(scores, items)
           if it["label"] == "ai" and not (s > thr)]
    fps = [(s, it) for s, it in zip(scores, items)
           if it["label"] == "human" and s > thr]
    if fns:
        print(f"\nfalse negatives (ai that would pass), {len(fns)}:")
        for s, it in sorted(fns, key=lambda x: -x[0]):
            print(f"  score={s:+.3f}  {it['text'][:88]}")
    if fps:
        print(f"\nfalse positives (human that would block), {len(fps)}:")
        for s, it in sorted(fps, key=lambda x: -x[0]):
            print(f"  score={s:+.3f}  {it['text'][:88]}")

    if json_out:
        json.dump({
            "n": len(items), "k": K, "threshold": thr,
            "metrics": base, "margin_only": margin_only,
            "threshold_sweep": {str(k): v for k, v in sweep.items()},
            "k_sweep": {str(k): v for k, v in ksweep.items()},
            "pending_embeddings": len(pending),
            "false_negatives": [it["text"] for _, it in fns],
            "false_positives": [it["text"] for _, it in fps],
        }, open(json_out, "w"), indent=1)
        print(f"\nwrote {json_out}")

    ok = base["precision"] >= GATE_P and base["recall"] >= GATE_R
    print(f"\nRESULT: {'GREEN' if ok else 'BLOCKED'} "
          f"(gate: precision >= {GATE_P}, recall >= {GATE_R})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
