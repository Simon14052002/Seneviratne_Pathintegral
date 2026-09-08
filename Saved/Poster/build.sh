#!/usr/bin/env bash
# Poster bauen und Vorschau erzeugen.
#   ./build.sh          -> poster.pdf + preview.png (150 dpi)
#   ./build.sh watch    -> baut bei jedem Speichern automatisch neu
set -e
cd "$(dirname "$0")"
if [ "$1" = "watch" ]; then
    exec latexmk -pdf -pvc -interaction=nonstopmode poster.tex
fi
if [ "$1" = "check" ]; then
    exec ./check_layout.py
fi
pdflatex -interaction=nonstopmode -file-line-error poster.tex | \
    grep -E "^(!|.*:[0-9]+:|Overfull \\\\hbox \([0-9]{2,})" || true
pdftoppm -png -r 150 -singlefile poster.pdf preview
./check_layout.py || true
echo "--> poster.pdf  +  preview.png"
