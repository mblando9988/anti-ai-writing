import json,os,re,subprocess,tempfile,sys,time
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK=os.path.join(ROOT,"bin","anti_ai_sem")
SRC=os.path.join(ROOT,"src","anti_ai_sem.swift")
CORPUS=json.load(open(os.path.join(ROOT,"corpus","anti_ai_corpus.json")))

def anti_drift():
    # The VERDICT must be embedding-only. Fail if the block guard depends on any
    # lexical signal, or if the hook loads a banned-word / wordnet data file.
    src=open(SRC).read()
    bad=[]
    # the line that calls block(...) must be guarded by aiVotes (k-NN), not lexical
    for ln in src.splitlines():
        if re.search(r"if .*\{\s*$",ln) and "margin >" in ln:
            if any(t in ln for t in ("struc","LEX","banned","stems","wordnet","rx(")):
                bad.append(f"verdict guard mixes lexical signal: {ln.strip()}")
            if "margin" not in ln:
                bad.append(f"verdict guard not embedding-based: {ln.strip()}")
    if not re.search(r"if\s+margin\s*>\s*MARGIN\s*\{", src):
        bad.append("no embedding margin block guard found")
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
a1=run(text="delve")!=2; a2=run(text="ensure the cache is enabled")!=2
a3=run(raw=b"")==0; a4=run(raw=b"\xff not json {{{")==0
fails=[]
for n,ok in [("delve allows",a1),("'ensure the cache is enabled' allows",a2),("empty fail-open",a3),("garbage fail-open",a4)]:
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
