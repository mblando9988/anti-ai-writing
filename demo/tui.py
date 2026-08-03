#!/usr/bin/env python3
# Terminal demo. Type a sentence, press Enter, see if it reads AI (red) or human
# (green) with the score and the nearest AI/human example. Ctrl-C to quit.
import json, os, subprocess, tempfile, curses, textwrap
from engine import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK, qvec, REF, ENGINE = load()

def _cos(a, b): return sum(x*y for x, y in zip(a, b))

def check(text):
    with tempfile.TemporaryDirectory() as td:
        tp = os.path.join(td, "t.jsonl")
        open(tp, "w").write(json.dumps({"type":"assistant","message":{"role":"assistant","content":text}})+"\n")
        p = subprocess.run(HOOK, input=json.dumps({"transcript_path":tp}).encode(), capture_output=True, timeout=60)
    why = p.stderr.decode().strip()
    reason = why.split("): ",1)[-1] if "):" in why else why.split("allow: ")[-1]
    q = qvec(text)
    ai = hu = []
    if q:
        ai = sorted(((_cos(q, v), t) for l, t, v in REF if l == "ai"), reverse=True)[:3]
        hu = sorted(((_cos(q, v), t) for l, t, v in REF if l == "human"), reverse=True)[:3]
    return p.returncode == 2, reason, ai, hu

SAMPLES = [
 "Honestly, that's a fantastic question — let me delve into the nuances to truly empower you.",
 "The bug is a null pointer on line 88. Add a guard before the deref and it goes away.",
]

def app(scr):
    curses.curs_set(1); scr.keypad(True)
    curses.start_color(); curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_RED, -1)     # AI
    curses.init_pair(2, curses.COLOR_GREEN, -1)   # human
    curses.init_pair(3, curses.COLOR_CYAN, -1)
    curses.init_pair(4, 8, -1)                     # dim
    buf = ""; result = None; status = "type a sentence (6+ words), Enter to check"
    while True:
        scr.erase(); h, w = scr.getmaxyx(); W = min(w-2, 90)
        scr.addstr(0, 1, "anti-ai-writing — terminal demo", curses.A_BOLD)
        scr.addstr(0, 35, "Ctrl-C quit · ↑ sample", curses.color_pair(4))
        scr.addstr(2, 1, "> ", curses.color_pair(3) | curses.A_BOLD)
        for i, line in enumerate(textwrap.wrap(buf, W-2) or [""]):
            scr.addstr(2+i, 3, line)
        y = 2 + max(1, len(textwrap.wrap(buf, W-2)))
        scr.addstr(y+1, 1, status, curses.color_pair(4))
        if result:
            flagged, reason, na, nh = result
            cp = curses.color_pair(1) if flagged else curses.color_pair(2)
            verdict = "  ⛔  READS AI-GENERATED  " if flagged else "  ✅  READS HUMAN  "
            scr.addstr(y+3, 1, verdict, cp | curses.A_BOLD | curses.A_REVERSE)
            scr.addstr(y+4, 1, reason[:W], curses.color_pair(4))
            best = max((na[0][0] if na else 0), (nh[0][0] if nh else 0))
            weak_thr = 0.88 if ENGINE == "apple" else 0.25  # hash cosines run lower
            weak = "   (weak — nothing close in corpus)" if best < weak_thr else ""
            scr.addstr(y+6, 1, "closest AI examples" + weak, curses.color_pair(1) | curses.A_BOLD)
            for i, (c, t) in enumerate(na):
                scr.addstr(y+7+i, 1, f"{c:.2f} ", curses.color_pair(4))
                scr.addstr(y+7+i, 6, t[:W-6])
            scr.addstr(y+11, 1, "closest human examples", curses.color_pair(2) | curses.A_BOLD)
            for i, (c, t) in enumerate(nh):
                scr.addstr(y+12+i, 1, f"{c:.2f} ", curses.color_pair(4))
                scr.addstr(y+12+i, 6, t[:W-6])
        scr.move(2, 3 + (len(buf) % (W-2)))
        scr.refresh()
        try: c = scr.get_wch()
        except curses.error: continue
        if c in ("\n", "\r", curses.KEY_ENTER):
            if len(buf.split()) >= 6:
                status = "checking…"; scr.addstr(y+1, 1, status); scr.refresh()
                result = check(buf); status = "type another, or Ctrl-C to quit"
            else:
                status = "needs 6+ words"
        elif c in (curses.KEY_BACKSPACE, "\x7f", "\b"): buf = buf[:-1]
        elif c == curses.KEY_UP: buf = SAMPLES[0] if buf != SAMPLES[0] else SAMPLES[1]
        elif isinstance(c, str) and c.isprintable(): buf += c

if __name__ == "__main__":
    try: curses.wrapper(app)
    except KeyboardInterrupt: pass
