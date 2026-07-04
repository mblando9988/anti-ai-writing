#!/usr/bin/env python3
# Engine picker for the demos. Uses the compiled macOS binaries when they're
# built (bin/anti_ai_sem + bin/embed_one over corpus_emb.json); otherwise falls
# back to the portable python hook, which embeds the corpus itself — so the
# demos work on Linux with nothing built at all.
import importlib.util, json, math, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def load():
    """Returns (hook_cmd, qvec, refs, name).
    hook_cmd: argv list that speaks the hook's stdin/exit-code contract.
    qvec(text) -> unit vector or None.  refs: [(label, text, unit vector)]."""
    hook_bin = os.path.join(ROOT, "bin", "anti_ai_sem")
    emb_bin = os.path.join(ROOT, "bin", "embed_one")
    if os.path.exists(hook_bin) and os.path.exists(emb_bin):
        raw = json.load(open(os.path.join(ROOT, "corpus", "corpus_emb.json")))
        refs = [(e["label"], e["text"], _unit(e["v"])) for e in raw]

        def qvec(text):
            out = subprocess.run([emb_bin], input=text.encode(),
                                 capture_output=True, timeout=60).stdout
            v = json.loads(out or "[]")
            return _unit(v) if len(v) > 10 else None

        return [hook_bin], qvec, refs, "apple"

    spec = importlib.util.spec_from_file_location(
        "anti_ai_sem", os.path.join(ROOT, "src", "anti_ai_sem.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    corpus = json.load(open(os.path.join(ROOT, "corpus", "anti_ai_corpus.json")))
    refs = [(x["label"], x["text"], mod.embed(x["text"])) for x in corpus]
    return ([sys.executable, os.path.join(ROOT, "src", "anti_ai_sem.py")],
            mod.embed, refs, "portable")
