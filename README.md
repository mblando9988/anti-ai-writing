# anti-ai-writing

Flags AI-sounding text by comparing it to labeled examples of AI and human writing.

It embeds a passage with Apple's NLContextualEmbedding and compares it with every example in the
corpus. The score is the average similarity to the 5 closest AI examples minus the average
similarity to the 5 closest human ones. A few phrase checks (praise openers, validation,
disagreement) nudge the score by a small fixed amount. Above 0.02, the text is flagged.

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

Runs every corpus item (489 right now) through the binary and prints precision and recall. `test/ab_eval.py` compares a plain k-NN vote with a keyword baseline on a held-out split. `test/eval.py` runs the case files under `test/cases/` and reports false positives and negatives.

## demo

Demos:

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

It appends the example and rebuilds the embeddings.

