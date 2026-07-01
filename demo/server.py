#!/usr/bin/env python3
import json, os, subprocess, tempfile, math, http.server, socketserver
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB=os.path.join(ROOT,"bin","embed_one"); HOOK=os.path.join(ROOT,"bin","anti_ai_sem")
ref=json.load(open(os.path.join(ROOT,"corpus","corpus_emb.json")))
def unit(v):
    n=math.sqrt(sum(x*x for x in v)) or 1.0; return [x/n for x in v]
def cos(a,b): return sum(x*y for x,y in zip(a,b))
def check(text):
    with tempfile.TemporaryDirectory() as td:
        tp=os.path.join(td,"t.jsonl"); open(tp,"w").write(json.dumps({"type":"assistant","message":{"role":"assistant","content":text}})+"\n")
        p=subprocess.run([HOOK],input=json.dumps({"transcript_path":tp}).encode(),capture_output=True,timeout=60)
    why=p.stderr.decode().strip()
    qv=json.loads(subprocess.run([EMB],input=text.encode(),capture_output=True,timeout=60).stdout or "[]")
    na=nh=None
    if len(qv)>10:
        q=unit(qv)
        ai=sorted(((cos(q,unit(e["v"])),e["text"]) for e in ref if e["label"]=="ai"),reverse=True)
        hu=sorted(((cos(q,unit(e["v"])),e["text"]) for e in ref if e["label"]=="human"),reverse=True)
        na=ai[0] if ai else None; nh=hu[0] if hu else None
    return {"flagged":p.returncode==2,"reason":why.split("): ",1)[-1] if "):" in why else why.split("allow: ")[-1],
            "near_ai":na,"near_human":nh}
class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self,*a,**k): super().__init__(*a,directory=os.path.join(ROOT,"demo"),**k)
    def do_POST(self):
        # oversized or unparseable requests get an error status — never a verdict
        # on text the hook didn't actually see
        try: n=int(self.headers.get("content-length",0) or 0)
        except ValueError: n=-1
        if n<0 or n>1_000_000:
            self.send_response(413 if n>1_000_000 else 400); self.end_headers(); return
        try: body=json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self.send_response(400); self.end_headers(); return
        t=body.get("text","") if isinstance(body,dict) else ""
        out=json.dumps(check(t if isinstance(t,str) else "")).encode()
        self.send_response(200); self.send_header("content-type","application/json"); self.send_header("content-length",str(len(out))); self.end_headers(); self.wfile.write(out)
    def log_message(self,*a): pass
if __name__=="__main__":
    missing=[p for p in (EMB,HOOK) if not os.path.exists(p)]
    if missing:
        print("missing binaries:",", ".join(missing))
        print("build them (macOS): swiftc -O src/anti_ai_sem.swift -o bin/anti_ai_sem"
              " && swiftc -O src/embed_one.swift -o bin/embed_one")
        raise SystemExit(3)
    with socketserver.TCPServer(("127.0.0.1",8778),H) as s:
        print("demo at http://127.0.0.1:8778"); s.serve_forever()
