# anti-ai-writing

Flags AI-sounding text by comparing it to labeled examples instead of matching keywords.

It embeds a passage with Apple's NLEmbedding (sentence vectors) and runs a k-nearest-neighbor
vote against a corpus of AI and human writing. If most of the nearest examples are AI, it flags
the text. There's no word list in the decision.

Two interchangeable engines, same contract (exit 2 = flagged, 0 = not; bad or
empty input exits 0):

- **portable** (`src/anti_ai_sem.py`): the default. Plain python3, no
  dependencies, any Linux or macOS. Two feature channels — hashed word/char
  n-grams (what's said) plus a structural style profile of punctuation, emoji,
  digits, casing, and sentence shape (how it's said; no word lists). It embeds
  the corpus itself at startup, so new examples work immediately — there is no
  re-embed step. Leave-one-out over the whole corpus: precision 1.000,
  recall 1.000.
- **macOS native** (`src/anti_ai_sem.swift`): Apple NLEmbedding sentence
  vectors — real semantics instead of surface features. Needs swiftc, and a
  corpus re-embed on a Mac when examples are added.

## run

anywhere (no build):

```
echo '{"transcript_path":"turn.jsonl"}' | python3 src/anti_ai_sem.py
```

macOS native (optional):

```
swiftc -O src/anti_ai_sem.swift -o bin/anti_ai_sem
echo '{"transcript_path":"turn.jsonl"}' | bin/anti_ai_sem
```

## test

```
python3 test/verify_sem.py
```

Runs every corpus item through the binary and prints precision/recall. Last run was 0.99/0.96 on 431 examples. `test/ab_eval.py` compares it against a plain keyword baseline on a held-out split. `test/eval.py` runs the case files under `test/cases/` and reports false positives and negatives.

## backtest & verify (no Mac needed)

```
python3 test/verify_portable.py      # every corpus example through the real hook
python3 test/backtest.py --portable  # leave-one-out replay + threshold/k sweeps
python3 test/backtest.py             # apple engine: precomputed embeddings
python3 test/lint_corpus.py          # data quality: dups, ids, embedding sync
HOOK="python3 src/anti_ai_sem.py" python3 test/eval.py   # 36-case suite
```

`verify_portable.py` is the end-to-end proof: it feeds each corpus example to
the hook exactly the way the runtime does (transcript file, stdin JSON, exit
codes) and checks the fail-open behaviors. The backtest replays each engine's
scoring math leave-one-out and prints threshold/k sweeps and every miss. Last
run — portable: precision 1.000, recall 1.000 on all 680 examples; apple:
precision 1.000, recall 0.970 on its 431 embedded. All of it runs in CI.

## demo

Two ways to see it decide, live:

```
# terminal UI: type a sentence, Enter to check, Ctrl-C to quit
python3 demo/tui.py

# web UI at http://127.0.0.1:8778
python3 demo/server.py
```

Both show the verdict, the score (embedding margin + positional nudge), and the
nearest AI and human examples it matched. On macOS with the binaries built
(`swiftc -O src/embed_one.swift -o bin/embed_one`) they use the native engine;
otherwise they fall back to the portable one and work anywhere.

## layout

```
src/      swift source + the corpus embedder
bin/      built binary (gitignored, build it yourself)
corpus/   labeled examples and their embeddings
test/     verifier + the a/b script
```

## teach it more

Add example sentences with one command (no editing files):

```
python3 add.py ai    "an AI-sounding sentence"
python3 add.py human "a normal human sentence"
```

It appends the example and rebuilds. Add a bunch and it gets better at the kinds you give it.
Pass `--no-rebuild` to batch several before one rebuild. On a machine without
swift it saves the example and tells you to rebuild on a Mac later — the
corpus currently has examples waiting for exactly that
(`python3 test/lint_corpus.py` shows the pending count; run
`swift ../src/embed_corpus.swift` from `corpus/` to fold them in).

