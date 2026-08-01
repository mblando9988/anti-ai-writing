#!/usr/bin/env python3
import json, os, subprocess, tempfile, http.server, socketserver
from engine import load
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK, qvec, REF, ENGINE = load()
def cos(a,b): return sum(x*y for x,y in zip(a,b))
def check(text):
    with tempfile.TemporaryDirectory() as td:
        tp=os.path.join(td,"t.jsonl"); open(tp,"w").write(json.dumps({"type":"assistant","message":{"role":"assistant","content":text}})+"\n")
        p=subprocess.run(HOOK,input=json.dumps({"transcript_path":tp}).encode(),capture_output=True,timeout=60)
    why=p.stderr.decode().strip()
    q=qvec(text)
    na=nh=None
    if q:
        ai=sorted(((cos(q,v),t) for l,t,v in REF if l=="ai"),reverse=True)
        hu=sorted(((cos(q,v),t) for l,t,v in REF if l=="human"),reverse=True)
        na=ai[0] if ai else None; nh=hu[0] if hu else None
    return {"flagged":p.returncode==2,"reason":why.split("): ",1)[-1] if "):" in why else why.split("allow: ")[-1],
            "near_ai":na,"near_human":nh}
class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self,*a,**k): super().__init__(*a,directory=os.path.join(ROOT,"demo"),**k)
    def do_POST(self):
        n=int(self.headers.get("content-length",0)); body=json.loads(self.rfile.read(n) or b"{}")
        out=json.dumps(check(body.get("text",""))).encode()
        self.send_response(200); self.send_header("content-type","application/json"); self.send_header("content-length",str(len(out))); self.end_headers(); self.wfile.write(out)
    def log_message(self,*a): pass
if __name__=="__main__":
    with socketserver.TCPServer(("127.0.0.1",8778),H) as s:
        print(f"demo at http://127.0.0.1:8778 ({ENGINE} engine)"); s.serve_forever()
