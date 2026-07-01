#!/usr/bin/env python3
# Terminal demo. Type a sentence, press Enter, see if it reads AI (red) or human
# (green) with the score and the nearest AI/human example. Ctrl-C to quit.
import json, os, subprocess, tempfile, math, curses, textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB  = os.path.join(ROOT, "bin", "embed_one")
HOOK = os.path.join(ROOT, "bin", "anti_ai_sem")
REF  = json.load(open(os.path.join(ROOT, "corpus", "corpus_emb.json")))

def _unit(v):
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]
def _cos(a, b): return sum(x*y for x, y in zip(a, b))

def check(text):
    with tempfile.TemporaryDirectory() as td:
        tp = os.path.join(td, "t.jsonl")
        open(tp, "w").write(json.dumps({"type":"assistant","message":{"role":"assistant","content":text}})+"\n")
        p = subprocess.run([HOOK], input=json.dumps({"transcript_path":tp}).encode(), capture_output=True, timeout=60)
    why = p.stderr.decode().strip()
    reason = why.split("): ",1)[-1] if "):" in why else why.split("allow: ")[-1]
    qv = json.loads(subprocess.run([EMB], input=text.encode(), capture_output=True, timeout=60).stdout or "[]")
    ai = hu = []
    if len(qv) > 10:
        q = _unit(qv)
        ai = sorted(((_cos(q,_unit(e["v"])), e["text"]) for e in REF if e["label"]=="ai"), reverse=True)[:3]
        hu = sorted(((_cos(q,_unit(e["v"])), e["text"]) for e in REF if e["label"]=="human"), reverse=True)[:3]
    return p.returncode == 2, reason, ai, hu

SAMPLES = [
 "Honestly, that's a fantastic question — let me delve into the nuances to truly empower you.",
 "The bug is a null pointer on line 88. Add a guard before the deref and it goes away.",
]

def safe(scr, y, x, s, attr=0):
    # never write past the screen: long input or a small terminal must not crash
    h, w = scr.getmaxyx()
    if 0 <= y < h - 1 and x < w - 1:
        try: scr.addstr(y, x, s[:w - x - 1], attr)
        except curses.error: pass

def app(scr):
    curses.curs_set(1); scr.keypad(True)
    curses.start_color(); curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_RED, -1)     # AI
    curses.init_pair(2, curses.COLOR_GREEN, -1)   # human
    curses.init_pair(3, curses.COLOR_CYAN, -1)
    curses.init_pair(4, 8, -1)                     # dim
    buf = ""; result = None; status = "type a sentence (6+ words), Enter to check"
    while True:
        scr.erase(); h, w = scr.getmaxyx(); W = max(3, min(w-2, 90))  # W-2 >= 1: wrap() and % need a positive width
        safe(scr, 0, 1, "anti-ai-writing — terminal demo", curses.A_BOLD)
        safe(scr, 0, 35, "Ctrl-C quit · ↑ sample", curses.color_pair(4))
        safe(scr, 2, 1, "> ", curses.color_pair(3) | curses.A_BOLD)
        for i, line in enumerate(textwrap.wrap(buf, W-2) or [""]):
            safe(scr, 2+i, 3, line)
        y = 2 + max(1, len(textwrap.wrap(buf, W-2)))
        safe(scr, y+1, 1, status, curses.color_pair(4))
        if result:
            flagged, reason, na, nh = result
            cp = curses.color_pair(1) if flagged else curses.color_pair(2)
            verdict = "  ⛔  READS AI-GENERATED  " if flagged else "  ✅  READS HUMAN  "
            safe(scr, y+3, 1, verdict, cp | curses.A_BOLD | curses.A_REVERSE)
            safe(scr, y+4, 1, reason[:W], curses.color_pair(4))
            best = max((na[0][0] if na else 0), (nh[0][0] if nh else 0))
            weak = "   (weak — nothing close in corpus)" if best < 0.88 else ""
            safe(scr, y+6, 1, "closest AI examples" + weak, curses.color_pair(1) | curses.A_BOLD)
            for i, (c, t) in enumerate(na):
                safe(scr, y+7+i, 1, f"{c:.2f} ", curses.color_pair(4))
                safe(scr, y+7+i, 6, t[:W-6])
            safe(scr, y+11, 1, "closest human examples", curses.color_pair(2) | curses.A_BOLD)
            for i, (c, t) in enumerate(nh):
                safe(scr, y+12+i, 1, f"{c:.2f} ", curses.color_pair(4))
                safe(scr, y+12+i, 6, t[:W-6])
        try: scr.move(min(2, h-2), min(3 + (len(buf) % (W-2)), w-2))
        except curses.error: pass
        scr.refresh()
        try: c = scr.get_wch()
        except curses.error: continue
        if c in ("\n", "\r", curses.KEY_ENTER):
            if len(buf.split()) >= 6:
                status = "checking…"; safe(scr, y+1, 1, status); scr.refresh()
                result = check(buf); status = "type another, or Ctrl-C to quit"
            else:
                status = "needs 6+ words"
        elif c in (curses.KEY_BACKSPACE, "\x7f", "\b"): buf = buf[:-1]
        elif c == curses.KEY_UP: buf = SAMPLES[0] if buf != SAMPLES[0] else SAMPLES[1]
        elif isinstance(c, str) and c.isprintable(): buf += c

if __name__ == "__main__":
    missing = [p for p in (EMB, HOOK) if not os.path.exists(p)]
    if missing:
        print("missing binaries:", ", ".join(missing))
        print("build them (macOS): swiftc -O src/anti_ai_sem.swift -o bin/anti_ai_sem"
              " && swiftc -O src/embed_one.swift -o bin/embed_one")
        raise SystemExit(3)
    try: curses.wrapper(app)
    except KeyboardInterrupt: pass
