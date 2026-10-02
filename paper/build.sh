#!/usr/bin/env bash
# Build all submission files: figures, manuscript, highlights and cover letter.
set -euo pipefail
cd "$(dirname "$0")"
python make_assets.py --results ../results/revision --out .
(cd figures && pdflatex -interaction=nonstopmode -halt-on-error fig1_pipeline.tex >/dev/null && rm -f fig1_pipeline.aux fig1_pipeline.log)
for doc in manuscript highlights cover_letter; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$doc.tex" >/dev/null
done
latexmk -c >/dev/null
echo "Built: manuscript.pdf highlights.pdf cover_letter.pdf"
