import json,os,re,math
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
corpus=json.load(open(os.path.join(ROOT,"corpus","anti_ai_corpus.json")))
emb=json.load(open(os.path.join(ROOT,"corpus","corpus_emb.json")))
vec={e["text"]:e["v"] for e in emb}
# frozen split: TEST = hand-authored/real sources; DEV(reference for B) = templated-generated.
REAL={"examples.md/before","examples.md/after","assistant-register","assistant-direct",
      "clean-dev","slop-seed","hedge-technical","blunt-technical-2"}
test=[c for c in corpus if c["source"] in REAL and c["text"] in vec]
dev =[c for c in corpus if c["source"] not in REAL and c["text"] in vec]
def metrics(items,pred):
    TP=FP=FN=TN=0
    for c in items:
        ai_true=c["label"]=="ai"; ai_pred=pred(c)
        if ai_true and ai_pred:TP+=1
        elif ai_true:FN+=1
        elif ai_pred:FP+=1
        else:TN+=1
    p=TP/(TP+FP) if TP+FP else 0; r=TP/(TP+FN) if TP+FN else 0
    f1=2*p*r/(p+r) if p+r else 0
    return dict(TP=TP,FP=FP,FN=FN,TN=TN,precision=round(p,3),recall=round(r,3),f1=round(f1,3))
# --- System A: lexical baseline (the abandoned keyword approach) ---
SLOP=set("delve leverage utilize robust seamless comprehensive pivotal crucial holistic nuanced multifaceted transformative innovative paramount vibrant tapestry realm landscape synergy harness unlock unleash showcase elevate empower streamline navigate foster underscore captivating compelling meticulous groundbreaking revolutionary honestly frankly absolutely certainly essentially fantastic brilliant excellent amazing wonderful incredible".split())
HEDGE=re.compile(r"^\W*(honestly|frankly|to be honest|absolutely|certainly|great question|i'?d be happy|happy to|of course)",re.I)
def A(c):
    t=c["text"].lower(); toks=set(re.findall(r"[a-z'\-]+",t)); slop=len(SLOP&toks)
    emdash="—" in c["text"]; tri=bool(re.search(r"\b\w+,\s+\w+,?\s+and\s+\w+",t))
    return slop>=2 or bool(HEDGE.match(c["text"])) or (emdash and slop>=1) or (tri and slop>=1)
# --- System B: semantic k-NN, reference = DEV only (honest held-out) ---
def cos(a,b):
    d=sum(x*y for x,y in zip(a,b)); 
    return d  # vectors are L2-normalized already
def B(c):
    q=vec[c["text"]]
    sims=sorted(((cos(q,vec[d["text"]]),d["label"]) for d in dev),reverse=True)[:5]
    return sum(1 for _,l in sims if l=="ai")>=3
print(f"split: TEST(real)={len(test)} (ai={sum(c['label']=='ai' for c in test)})  DEV/ref(generated)={len(dev)}")
print("System A (lexical baseline):", metrics(test,A))
print("System B (semantic k-NN, dev-only ref):", metrics(test,B))
