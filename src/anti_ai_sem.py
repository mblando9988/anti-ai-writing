#!/usr/bin/env python3
# The Linux (portable) variant of the hook. Same contract as anti_ai_sem.swift:
# stdin gets {"transcript_path": ...} (or {"_text": ...}), exit 2 means flagged,
# exit 0 means not. Short, empty, or broken input exits 0 (fail-open).
#
# Same pipeline too: embed the passage, take the top-k mean cosine similarity
# to the AI examples minus the top-k mean to the human ones (the contrastive
# margin), add the same bounded positional nudge, gate on a threshold.
#
# Only the embedder differs. Apple's NLEmbedding doesn't exist off macOS, so
# this uses hashed word/bigram/char-n-gram features (the hashing trick, no
# dependencies, no model download) and vectorizes the corpus straight from
# corpus/anti_ai_corpus.json at startup — every corpus example works here the
# moment it's added, no re-embed step. The feature space is different, so the
# engine has its own k and threshold, calibrated by leave-one-out backtest
# (python3 test/backtest.py --portable).
#
# Env: SR_MARGIN overrides the threshold, same as the Swift hook.
import json, math, os, re, sys
from hashlib import blake2b

DIM = 2048
K = 9
DEFAULT_MARGIN = 0.025
STYLE_W = 0.10   # weight of the structural style channel in the similarity


def _ngram_vec(text):
    """Hashed n-gram channel: words, word bigrams, char 3-5 grams, signed
    hashing into DIM buckets, sublinear tf, L2-normalized."""
    t = re.sub(r"\s+", " ", text.lower().strip())
    words = re.findall(r"[a-z0-9']+|[^\sa-z0-9']", t)
    f = {}

    def add(k):
        f[k] = f.get(k, 0) + 1

    for w in words:
        add("w:" + w)
    for a, b in zip(words, words[1:]):
        add("b:" + a + "_" + b)
    s = " " + t + " "
    for n in (3, 4, 5):
        for i in range(len(s) - n + 1):
            add("c%d:" % n + s[i:i + n])
    v = [0.0] * DIM
    for k, tf in f.items():
        d = blake2b(k.encode(), digest_size=8).digest()
        u = int.from_bytes(d, "little")
        v[(u >> 1) % DIM] += (1.0 if u & 1 else -1.0) * (1 + math.log(tf))
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _style_vec(text):
    """Structural style channel: punctuation, emoji, digits, casing, and
    sentence-shape statistics. No word lists — register, not topic. This is
    what lets 'Proud beyond words of this incredible team!' sit next to
    'Thrilled to announce our journey…' when they share no n-grams, and what
    keeps terse, digit-heavy, code-flavored human replies away from slop."""
    t = text.strip()
    words = re.findall(r"\S+", t)
    W = max(len(words), 1)
    lw = re.findall(r"[A-Za-z]+", t)
    sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
    f = [
        t.count("!") / W * 8,
        1.0 if re.search(r"[\U0001F000-\U0001FAFF✀-➿☀-⛿✨❤]", t) else 0.0,
        sum(1 for w in words if re.search(r"\d", w)) / W * 4,
        sum(1 for w in words if re.search(r"[/_]|--|\(\)|\.\w", w)) / W * 4,
        1.0 if re.search(r"\d+\s?(ms|s\b|gb|mb|kb|%|x\b|min\b)", t, re.I) else 0.0,
        min(sum(len(w) for w in lw) / max(len(lw), 1), 9) / 9,
        t.count(",") / W * 6,
        t.count(";") / W * 8,
        t.count(":") / W * 8,
        (t.count("—") + t.count(" - ")) / W * 8,
        t.count("?") / W * 8,
        1.0 if t[:1].islower() else 0.0,
        min(W / max(len(sents), 1), 30) / 30,
        sum(1 for w in words if "'" in w) / W * 6,
        1.0 if re.search(r"[\"“”`]", t) else 0.0,
        1.0 if "(" in t else 0.0,
        sum(1 for w in words[1:] if re.match(r"[A-Z][a-z]", w)) / W * 4,
        sum(1 for w in words if re.match(r"[A-Z]{2,}$", re.sub(r"\W", "", w))) / W * 6,
        min(W, 40) / 40,
        sum(1 for w in words if "-" in w and not w.startswith("-")) / W * 6,
    ]
    norm = math.sqrt(sum(x * x for x in f)) or 1.0
    return [x / norm for x in f]


def embed(text):
    """Two channels concatenated with sqrt weights, so the dot product of two
    embeddings is exactly (1-w)*cos_ngram + w*cos_style, and the vector stays
    unit-norm."""
    a = math.sqrt(1 - STYLE_W)
    b = math.sqrt(STYLE_W)
    return ([x * a for x in _ngram_vec(text)] +
            [x * b for x in _style_vec(text)])


def nudge(text, engine="portable"):
    """The positional nudge, same regexes and weights as anti_ai_sem.swift.
    Returns (value, reasons).

    One engine split: the bare-adverb opener bonus ("honestly", "frankly", ...)
    exists to give the Apple embedder back the word position its mean-pooling
    discards. The hashed n-gram space never loses surface openers — they're
    features already — so scoring them again there double-counts and flags
    honest human frustration ("honestly, this is broken"). The portable engine
    keeps only the explicit-praise opener, which is praise by construction."""
    lower = text.lower()
    opener_praise = bool(re.match(
        r"^\W*(great|excellent|fantastic|brilliant|wonderful|amazing|perfect)"
        r"\s+(question|point|observation)", text, re.I))
    if engine == "apple" and not opener_praise:
        opener_praise = bool(re.match(
            r"^\W*(honestly|frankly|absolutely|certainly|of course)\b", text[:120].lower()))
    validation = bool(re.search(
        r"your (theory|hypothesis|framing|instinct|intuition|perspective|premise) (is|sounds)"
        r"|you raise (a )?(great|good|valid|important) point"
        r"|completely (valid|understandable)|spot on|on the right track"
        r"|you'?re (absolutely )?right", lower))
    # "you're not wrong" is validation wearing a negation costume — it praises,
    # it doesn't push back. Count it as validation and keep its "not wrong"
    # away from the disagreement redemption below.
    if re.search(r"you'?re not wrong|you are not wrong", lower):
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


def load_corpus():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "..", "corpus", "anti_ai_corpus.json"),
              os.path.join(here, "anti_ai_corpus.json")):
        if os.path.exists(p):
            return json.load(open(p))
    return []


def margin(text, refs, k=K):
    """Contrastive margin of `text` against embedded refs
    [(label, text, vector), ...], skipping exact-text self matches."""
    q = embed(text)
    ai, hu = [], []
    for label, rtext, rv in refs:
        if rtext == text:
            continue
        c = sum(a * b for a, b in zip(q, rv))  # vectors are unit-norm
        (ai if label == "ai" else hu).append(c)
    ai.sort(reverse=True)
    hu.sort(reverse=True)
    top_ai = ai[:k] or [0.0]
    top_hu = hu[:k] or [0.0]
    return sum(top_ai) / len(top_ai) - sum(top_hu) / len(top_hu)


def score(text, refs, k=K):
    m = margin(text, refs, k)
    n, why = nudge(text)
    return m + n, m, n, why


def last_assistant_text(path):
    try:
        content = open(path, encoding="utf-8").read()
    except OSError:
        return ""
    last = ""
    for line in content.split("\n"):
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if not isinstance(o, dict):
            continue
        m = o.get("message") or {}
        if o.get("type") != "assistant" and m.get("role") != "assistant":
            continue
        c = m.get("content")
        if isinstance(c, str):
            last = c
        elif isinstance(c, list):
            t = " ".join(p.get("text", "") for p in c
                         if isinstance(p, dict) and p.get("type") == "text")
            if t.strip():
                last = t
    return last


def main():
    def allow(why=""):
        if why:
            print(f"[anti-ai-sem] allow: {why}", file=sys.stderr)
        sys.exit(0)

    def block(why):
        print(f"BLOCKED by anti-ai-sem (reads as AI-generated): {why}", file=sys.stderr)
        sys.exit(2)

    raw = sys.stdin.buffer.read()
    if not raw:
        allow("empty stdin")
    try:
        obj = json.loads(raw)
    except ValueError:
        allow("invalid stdin")
    if not isinstance(obj, dict):
        allow("invalid stdin")
    # if we already blocked once this turn, don't block again — avoids the re-block loop
    if obj.get("stop_hook_active") is True:
        allow("stop_hook_active")

    text = ""
    if isinstance(obj.get("transcript_path"), str):
        text = last_assistant_text(obj["transcript_path"])
    if not text and isinstance(obj.get("_text"), str):
        text = obj["_text"]
    text = text.strip()
    if not text:
        allow("empty turn")
    if len([w for w in re.split(r"[ \n]", text) if w]) < 6:
        allow("too short to judge style")

    corpus = load_corpus()
    if not corpus:
        allow("no reference corpus")
    refs = [(x["label"], x["text"], embed(x["text"])) for x in corpus]

    try:
        thr = float(os.environ.get("SR_MARGIN", ""))
    except ValueError:
        thr = DEFAULT_MARGIN

    s, m, n, why = score(text, refs)
    detail = (f"score={s:.2f} (margin={m:.2f}, nudge={n:+.2f} "
              f"{','.join(why) if why else '—'}) [portable engine]")
    if s > thr:
        block(detail)
    allow(detail + " — reads human")


if __name__ == "__main__":
    main()
