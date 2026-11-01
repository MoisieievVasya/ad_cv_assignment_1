"""Turn a percent-format script (`# %%` code cells, `# %% [markdown]` text cells) into an executed .ipynb.

    uv run python scripts/build_notebook.py notebooks/src/00_eda_metric_validation.py

The .py source is the reviewed artefact (clean diffs); the .ipynb next to `notebooks/` keeps the outputs.
"""
import re
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]


def parse(text):
    cells = []
    for block in re.split(r"^# %%", text, flags=re.M)[1:]:
        header, _, body = block.partition("\n")
        body = body.strip("\n")
        if "[markdown]" in header:
            md = "\n".join(re.sub(r"^# ?", "", line) for line in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(md))
        elif body:
            cells.append(nbformat.v4.new_code_cell(body))
    return cells


def main(src):
    src = Path(src)
    nb = nbformat.v4.new_notebook(cells=parse(src.read_text()))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = ROOT / "notebooks" / (src.stem + ".ipynb")
    ExecutePreprocessor(timeout=None, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT / "notebooks")}})
    nbformat.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
