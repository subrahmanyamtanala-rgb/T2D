"""Create a flat LaTeX package for Elsevier Editorial Manager (which flattens folders).

Inlines every \\input, drops \\graphicspath, copies figure PDFs, the .bib and the .bbl
into paper/submission/, and zips it as paper/submission/cbm_latex_source.zip.
"""

import re
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "submission"


def inline(path):
    text = path.read_text()

    def repl(m):
        target = HERE / m.group(1)
        if target.suffix != ".tex":
            target = target.with_suffix(".tex")
        return inline(target).rstrip("\n")

    return re.sub(r"\\input\{([^}]+)\}", repl, text)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    tex = inline(HERE / "manuscript.tex")
    tex = tex.replace("\\graphicspath{{figures/}}\n", "")
    (OUT / "manuscript.tex").write_text(tex)
    figures = sorted(set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)))
    for f in figures:
        shutil.copy(HERE / "figures" / f, OUT / f)
    for f in ("references.bib", "manuscript.bbl"):
        shutil.copy(HERE / f, OUT / f)
    with zipfile.ZipFile(OUT / "cbm_latex_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            if f.suffix != ".zip":
                z.write(f, f.name)
    print(f"submission package: {OUT / 'cbm_latex_source.zip'} ({len(figures)} figures)")


if __name__ == "__main__":
    main()
