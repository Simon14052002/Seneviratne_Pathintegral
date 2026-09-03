#!/usr/bin/env bash
# Hilfsmessung: baut eine Kopie OHNE Fussbalken, damit man sieht, wie weit
# die Spalten wirklich reichen, wenn sie unten ueberlaufen.
set -e
cd "$(dirname "$0")"
D=$(mktemp -d)
sed 's/\\fill\[jkufoot\]/\\fill[pagebg]/; s/text=white,font={\\fontsize{28}/text=pagebg,font={\\fontsize{28}/' poster.tex > "$D/free.tex"
pdflatex -interaction=nonstopmode -output-directory="$D" "$D/free.tex" >/dev/null 2>&1 || true
python3 - "$D/free.pdf" <<'PY'
import subprocess,sys,tempfile,os
pdf=sys.argv[1]; DPI=50
with tempfile.TemporaryDirectory() as td:
    st=os.path.join(td,"m")
    subprocess.run(["pdftoppm","-gray","-r",str(DPI),"-singlefile",pdf,st],check=True)
    b=open(st+".pgm","rb").read()
i,vals=2,[]
while len(vals)<3:
    while b[i:i+1].isspace(): i+=1
    if b[i:i+1]==b'#':
        while b[i:i+1]!=b'\n': i+=1
        continue
    j=i
    while not b[j:j+1].isspace(): j+=1
    vals.append(int(b[i:j])); i=j
W,H,px=vals[0],vals[1],b[i+1:]
mm=25.4/DPI
SIDE,SEP=25.0,9.0
colw=(W*mm-2*SIDE-2*SEP)/3
BG=px[int(0.5*H)*W+int((SIDE+colw+SEP/2)/mm)]
TARGET=1138.1
x=SIDE
for k in range(3):
    x0,x1=int(x/mm)+4,int((x+colw)/mm)-4; x+=colw+SEP
    last=0
    for y in range(int(0.10*H),H):
        if any(abs(px[y*W+xx]-BG)>4 for xx in range(x0,x1,3)): last=y
    print(f"Spalte {k+1}: Unterkante {last*mm:7.1f} mm   -> {last*mm-TARGET:+7.1f} mm")
PY
rm -rf "$D"
