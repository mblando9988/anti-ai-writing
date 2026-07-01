# anti-ai-writing

Flags AI-sounding text by comparing it to labeled examples instead of matching keywords.

It embeds a passage with Apple's NLContextualEmbedding (mean-pooled token vectors) and scores
a contrastive margin against a corpus of AI and human writing: how much closer the passage sits
to its five nearest AI examples than to its five nearest human ones, plus a small positional
nudge for openers the pooled vector can't see. If the score clears a threshold, it flags the
text. There's no word list in the decision.

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

Runs every corpus item through the binary and prints precision/recall (needs the binary, so macOS). The rest of the suite:

- `test/test_contract.py` — pins the input contract: transcript parsing (last assistant
  message, array content, tool_use turns, malformed lines), fail-open guards,
  `SR_MARGIN`, the 6-word floor, and the stderr format the demos parse. Needs the
  binary; `--mock` runs the same fixtures against `test/mock_hook.py` anywhere.
- `test/score_replay.py` — replays the verdict math (margin + nudge) over the frozen
  embeddings; runs anywhere, no binary. A parity gate fails if its ported regexes
  drift from `src/anti_ai_sem.swift`. Currently 1.00 precision / 0.995 recall
  leave-one-out over the 425 embedded examples.
- `test/eval.py` — runs the held-out case files under `test/cases/` (18 positive,
  18 negative, 12 edge) and reports false positives and negatives (macOS).
- `test/ab_eval.py` — compares against a plain keyword baseline on a held-out split.
- `test/check_dataset.py` — corpus hygiene: duplicates, labels, embedding sync.

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

It appends the example and rebuilds (`--no-rebuild` to skip the rebuild). It refuses
examples under 6 words — the hook never judges text that short — and skips duplicates.
Add a bunch and it gets better at the kinds you give it.

The corpus (`corpus/anti_ai_corpus.json`, 609 examples) can carry items that aren't embedded
yet — the hook only consults `corpus_emb.json`, and unembedded items still work as eval
queries in `test/verify_sem.py`. To turn them into reference examples, rebuild on a Mac:

```
cd corpus && swift ../src/embed_corpus.swift
```

`test/check_dataset.py` tells you how many items are pending embedding.

