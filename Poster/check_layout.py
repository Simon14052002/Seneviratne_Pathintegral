#!/usr/bin/env python3
"""Misst die Blockkanten im fertigen poster.pdf und meldet,
ob die Abstaende gleich sind und die Spalten buendig enden.

Aufruf:  ./check_layout.py       (oder  ./build.sh check)

Nuetzlich, sobald du die Dummy-Texte durch echte ersetzt: der Inhalt
wird laenger oder kuerzer, dann stimmt die Balance nicht mehr.
Nachjustieren dann ueber die Hoehen der \\dummyfig/\\includegraphics
oder indem ein Block in eine andere Spalte wandert.
"""
import subprocess, sys, os, tempfile

# --- muessen zu den Werten in poster.tex passen -----------------------
# Alle Masse werden aus poster.tex gelesen -- so koennen sie dort geaendert
# werden, ohne dass diese Pruefung veraltet.
import re as _re
_tex = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "poster.tex")).read()
def _mm(name, default):
    m = _re.search(r"\\setlength\{\\%s\}\s*\{([0-9.]+)mm\}" % name, _tex)
    return float(m.group(1)) if m else default
def _frac(name, default):
    m = _re.search(r"\\setlength\{\\%s\}\s*\{([0-9.]+)\\paperheight\}" % name, _tex)
    return float(m.group(1)) if m else default
SIDEMARGIN_MM = _mm("sidemargin", 25.0)
COLSEP_MM     = _mm("colsep", 9.0)
TOPGAP_MM     = _mm("topgap", 25.0)
BOTGAP_MM     = _mm("botgap", 25.0)
BLOCKSKIP_MM  = _mm("blockskip", 9.0)
HEADERH, FOOTH = _frac("headerh", 0.072), _frac("footh", 0.022)
DPI = 50

here = os.path.dirname(os.path.abspath(__file__))
pdf = os.path.join(here, "poster.pdf")

def read_pgm(path):
    b = open(path, "rb").read()
    assert b[:2] == b"P5", "kein PGM"
    i, vals = 2, []
    while len(vals) < 3:
        while b[i:i+1].isspace(): i += 1
        if b[i:i+1] == b"#":
            while b[i:i+1] != b"\n": i += 1
            continue
        j = i
        while not b[j:j+1].isspace(): j += 1
        vals.append(int(b[i:j])); i = j
    return vals[0], vals[1], b[i+1:]

with tempfile.TemporaryDirectory() as td:
    stem = os.path.join(td, "m")
    subprocess.run(["pdftoppm", "-gray", "-r", str(DPI), "-singlefile", pdf, stem], check=True)
    W, H, px = read_pgm(stem + ".pgm")

mm = 25.4 / DPI
page_mm = H * mm
top_target = HEADERH * page_mm + TOPGAP_MM
bot_target = page_mm - FOOTH * page_mm - BOTGAP_MM

# Hintergrundton aus dem Zwischenraum zwischen Spalte 1 und 2 ablesen;
# "Block" heisst dann einfach: weicht vom Hintergrund ab.
GX = int((SIDEMARGIN_MM + (W*mm - 2*SIDEMARGIN_MM - 2*COLSEP_MM)/3 + COLSEP_MM/2) / mm)
BG = px[int(0.5*H)*W + GX]

def blocks(x0, x1):
    ink = [any(abs(px[y*W + x] - BG) > 4 for x in range(x0, x1, 3)) for y in range(H)]
    segs, s = [], None
    for y in range(H):
        if ink[y] and s is None: s = y
        elif not ink[y] and s is not None:
            segs.append((s, y-1)); s = None
    if s is not None: segs.append((s, H-1))
    # Kopf- und Fussbalken ausblenden: es zaehlt, wo ein Block BEGINNT.
    # Laeuft ein Block unten ueber, bleibt er drin -- genau das soll die
    # Meldung ja zeigen.
    return [g for g in segs
            if g[1]-g[0] > 3
            and (HEADERH+0.008)*H < g[0] < (1-FOOTH-0.006)*H]

print(f"Seite {W/ (DPI/25.4):.0f} x {page_mm:.0f} mm    "
      f"Sollwerte: Oberkante {top_target:.1f} mm, Unterkante {bot_target:.1f} mm, "
      f"Blockabstand {BLOCKSKIP_MM:.1f} mm\n")

bad = False
page_w_mm = W * mm
colw_mm = (page_w_mm - 2*SIDEMARGIN_MM - 2*COLSEP_MM) / 3
x = SIDEMARGIN_MM
for k in range(3):
    x0, x1 = int(x/mm)+4, int((x+colw_mm)/mm)-4
    x += colw_mm + COLSEP_MM
    bs = blocks(x0, x1)
    print(f"Spalte {k+1}:  {len(bs)} Bloecke")
    prev = None
    for a, b in bs:
        gap = "" if prev is None else f"    Abstand {((a-prev-1)*mm):5.1f} mm"
        if prev is not None and abs((a-prev-1)*mm - BLOCKSKIP_MM) > 1.0:
            gap += "  <-- ungleich!"; bad = True
        print(f"   y {a*mm:7.1f} - {b*mm:7.1f} mm   Hoehe {(b-a+1)*mm:6.1f} mm{gap}")
        prev = b
    if bs:
        dtop, dbot = bs[0][0]*mm - top_target, bs[-1][1]*mm - bot_target
        flag = ""
        if abs(dtop) > 1.5 or abs(dbot) > 1.5:
            flag = "  <-- nicht buendig!"; bad = True
        print(f"   Oberkante {bs[0][0]*mm:.1f} mm ({dtop:+.1f}), "
              f"Unterkante {bs[-1][1]*mm:.1f} mm ({dbot:+.1f}){flag}\n")

print("Layout ok." if not bad else "Bitte nachjustieren (siehe Markierungen).")
sys.exit(1 if bad else 0)
