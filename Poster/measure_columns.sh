#!/usr/bin/env bash
# Misst die NATUERLICHE Hoehe jeder Spalte, indem LaTeX sie selbst ausgibt.
# Funktioniert auch dann noch, wenn eine Spalte weit ueber die Seite
# hinauslaeuft -- anders als das Ausmessen am gerenderten Bild.
set -e
cd "$(dirname "$0")"
D=$(mktemp -d)
python3 - "$D/mess.tex" <<'PY'
import sys, re
s = open('poster.tex').read()
hook = r"""
\newsavebox{\poscolbox}
\renewenvironment{postercolumn}
  {\setbox\poscolbox=\hbox\bgroup\begin{minipage}[t]{\colw}%
     \setlength{\textwidth}{\colw}\setlength{\columnwidth}{\colw}}
  {\end{minipage}\egroup
   \typeout{COLHEIGHT \the\ht\poscolbox\space \the\dp\poscolbox}%
   \box\poscolbox}
\begin{document}"""
# zusaetzlich jeden Blockkoerper vermessen
s = s.replace(r"  \par\vskip1.2ex\egroup",
              "  \\par\\vskip1.2ex\\egroup\n  \\typeout{BLOCK \\the\\ht0\\space \\the\\dp0}%")
s = s.replace(r"\begin{document}", hook, 1)
open(sys.argv[1], 'w').write(s)
PY
pdflatex -interaction=nonstopmode -output-directory="$D" "$D/mess.tex" >/dev/null 2>&1 || true
python3 - "$D/mess.log" <<'PY'
import sys, re, os
PT_MM = 25.4/72.27

# Die Masse werden aus poster.tex gelesen, damit sie nicht auseinanderlaufen,
# wenn dort etwas geaendert wird.
tex = open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])) or '.', 'poster.tex')).read() \
      if False else open('poster.tex').read()
def mm(name, default):
    m = re.search(r"\\setlength\{\\%s\}\s*\{([0-9.]+)mm\}" % name, tex)
    return float(m.group(1)) if m else default
def frac(name, default):
    m = re.search(r"\\setlength\{\\%s\}\s*\{([0-9.]+)\\paperheight\}" % name, tex)
    return float(m.group(1)) if m else default
PAGE_H = 1189.2
TOPGAP, BOTGAP = mm('topgap', 25.0), mm('botgap', 25.0)
HEADERH, FOOTH = frac('headerh', 0.072), frac('footh', 0.022)
TARGET = (PAGE_H - FOOTH*PAGE_H - BOTGAP) - (HEADERH*PAGE_H + TOPGAP)
log = open(sys.argv[1], errors='ignore').read()
vals = re.findall(r"COLHEIGHT ([0-9.]+)pt ([0-9.]+)pt", log)
if os.environ.get("BLOCKS"):
    blocks = re.findall(r"BLOCK ([0-9.]+)pt ([0-9.]+)pt", log)
    print("Blockkoerper (ohne Titelbalken), in Reihenfolge des Dokuments:")
    for n, (h, d) in enumerate(blocks, 1):
        print(f"   Block {n:2d}: {(float(h)+float(d))*25.4/72.27:7.1f} mm")
    print()
if not vals:
    print("keine Messwerte -- Kompilierfehler?"); raise SystemExit(1)
print(f"Verfuegbare Spaltenhoehe: {TARGET:.1f} mm  "
      f"(topgap {TOPGAP:.0f} mm, botgap {BOTGAP:.0f} mm -- aus poster.tex)\n")
for k, (h, d) in enumerate(vals[:3], 1):
    mm = (float(h)+float(d))*PT_MM
    print(f"Spalte {k}: {mm:8.1f} mm   {mm-TARGET:+7.1f} mm")
PY
rm -rf "$D"
