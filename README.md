# anti-ai-writing

Flags AI-sounding text by comparing it to labeled examples instead of matching keywords.

It embeds a passage with Apple's NLEmbedding (sentence vectors) and runs a k-nearest-neighbor
vote against a corpus of AI and human writing. If most of the nearest examples are AI, it flags
the text. There's no word list in the decision.

macOS only (uses the NaturalLanguage framework). Needs swiftc and python3.

## build

```
swiftc -O src/anti_ai_sem.swift -o bin/anti_ai_sem
```

## run

```
echo '{"transcript_path":"turn.jsonl"}' | bin/anti_ai_sem
```

Exit 2 means flagged, 0 means not. Bad or empty input exits 0.

## test

```
python3 test/verify_sem.py
```

Runs every corpus item through the binary and prints precision/recall. Last run was 0.99/0.96 on 431 examples. `test/ab_eval.py` compares it against a plain keyword baseline on a held-out split. `test/eval.py` runs the case files under `test/cases/` and reports false positives and negatives.

## backtest (no Mac needed)

```
python3 test/backtest.py       # leave-one-out over the precomputed embeddings
python3 test/lint_corpus.py    # data quality: dups, ids, embedding sync
```

The backtest replays the hook's exact scoring math (top-5 cosine margin plus
the positional nudge) over `corpus/corpus_emb.json`, leaving each example out
of its own vote. It prints precision/recall, a threshold sweep, a k sweep, and
every miss. Last run: precision 1.000, recall 0.970 on the 431 embedded
examples. It can't embed *new* text (that still needs macOS), but it means the
scoring and the data can be evaluated and tuned anywhere. Both run in CI.

## demo

Two ways to see it decide, live:

```
# terminal UI: type a sentence, Enter to check, Ctrl-C to quit
python3 demo/tui.py

# web UI at http://127.0.0.1:8778
python3 demo/server.py
```

Both show the verdict, the score (embedding margin + positional nudge), and the
nearest AI and human examples it matched. They need `bin/embed_one` built:
`swiftc -O src/embed_one.swift -o bin/embed_one`.

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

