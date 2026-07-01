import json,os,re,subprocess,tempfile,sys,time
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK=os.path.join(ROOT,"bin","anti_ai_sem")
SRC=os.path.join(ROOT,"src","anti_ai_sem.swift")
if not os.path.exists(HOOK):
    print("bin/anti_ai_sem not built (macOS: swiftc -O src/anti_ai_sem.swift -o bin/anti_ai_sem)");sys.exit(3)
CORPUS=json.load(open(os.path.join(ROOT,"corpus","anti_ai_corpus.json")))

def anti_drift():
    # Hybrid contract: the embedding margin is the BASE of the score; a small
    # positional nudge may adjust it. Fail if the embedding base is missing or
    # the score is not built on it, or if the hook loads a word-list data file.
    src=open(SRC).read()
    bad=[]
    if not re.search(r"let\s+margin\s*=\s*aiMean\s*-\s*huMean", src):
        bad.append("no embedding margin (aiMean - huMean)")
    if not re.search(r"let\s+score\s*=\s*margin\s*\+\s*nudge", src):
        bad.append("score not based on the embedding margin")
    if not re.search(r"if\s+score\s*>\s*MARGIN\s*\{", src):
        bad.append("verdict not gated on score>MARGIN")
    if re.search(r"stop_banned_words|wordnet_synonyms", src):
        bad.append("hook loads a lexical word-list data file")
    return bad
def run(text=None,raw=None):
    if raw is not None:
        return subprocess.run([HOOK],input=raw,capture_output=True,timeout=60).returncode
    with tempfile.TemporaryDirectory() as td:
        tp=os.path.join(td,"t.jsonl");open(tp,"w").write(json.dumps({"type":"assistant","message":{"role":"assistant","content":text}})+"\n")
        return subprocess.run([HOOK],input=json.dumps({"transcript_path":tp}).encode(),capture_output=True,timeout=60).returncode
t0=time.time();TP=FP=FN=TN=0;fp=[];fn=[]
for c in CORPUS:
    blocked=run(text=c["text"])==2
    if c["label"]=="ai":
        if blocked:TP+=1
        else:FN+=1;fn.append(c)
    else:
        if blocked:FP+=1;fp.append(c)
        else:TN+=1
prec=TP/(TP+FP) if TP+FP else 0; rec=TP/(TP+FN) if TP+FN else 0
print(f"REAL HOOK over {len(CORPUS)} items in {time.time()-t0:.0f}s")
print(f"TP={TP} FP={FP} FN={FN} TN={TN}  precision={prec:.3f} recall={rec:.3f}  (need >=0.90 / >=0.85)")
# note: these two are floor checks — 1 and 5 words never reach the classifier.
# the "single buzzword in a real sentence still passes" property is covered by
# the >=6-word delve-logs case in test/cases/edge/.
a1=run(text="delve")==0; a2=run(text="ensure the cache is enabled")==0
a3=run(raw=b"")==0; a4=run(raw=b"\xff not json {{{")==0
fails=[]
for n,ok in [("1-word input allows (6-word floor)",a1),("5-word input allows (6-word floor)",a2),("empty fail-open",a3),("garbage fail-open",a4)]:
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}");  fails.append(n) if not ok else None
print("-- false positives (human blocked) --")
for c in fp[:10]: print(f"  FP {c['id']} [{c['source']}] {c['text'][:75]}")
print("-- false negatives (ai allowed) --")
for c in fn[:10]: print(f"  FN {c['id']} [{c['source']}] {c['text'][:75]}")
drift=anti_drift()
print("-- anti-lexical-drift (verdict must be embedding-only) --")
print(f"  [{'PASS' if not drift else 'FAIL'}] semantic verdict" + ("" if not drift else ": "+"; ".join(drift)))
green=prec>=0.90 and rec>=0.85 and not fails and not drift
print("\nRESULT:", "GREEN" if green else "BLOCKED"); sys.exit(0 if green else 1)
