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

