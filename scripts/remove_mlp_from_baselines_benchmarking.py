"""Remove the MLP baseline from notebooks/2_Baselines_Benchmarking.ipynb.

The notebook is generated from scripts/generate_baselines_benchmarking_notebook.py,
so this script first checks that the generator has been updated to exclude MLP,
then regenerates the notebook. LSTM, RNN, and Transformer are preserved.
"""

from __future__ import annotations

import ast
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "scripts" / "generate_baselines_benchmarking_notebook.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "2_Baselines_Benchmarking.ipynb"


REQUIRED_MODELS = ("LSTMClassifier", "RNNClassifier", "TransformerClassifier")
FORBIDDEN_MARKERS = ("MLPClassifier", '"MLP"', "'MLP'", "make_mlp_loader", "mlp_hidden")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def validate_generator() -> None:
    text = read_text(GENERATOR_PATH)
    missing = [marker for marker in REQUIRED_MODELS if marker not in text]
    forbidden = [marker for marker in FORBIDDEN_MARKERS if marker in text]

    if missing:
        raise RuntimeError(f"Generator is missing required models: {missing}")
    if forbidden:
        raise RuntimeError(f"Generator still contains MLP markers: {forbidden}")


def validate_notebook() -> None:
    notebook = json.loads(read_text(NOTEBOOK_PATH))
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]

    for cell in code_cells:
        ast.parse("".join(cell.get("source", [])))

    notebook_text = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    missing = [marker for marker in REQUIRED_MODELS if marker not in notebook_text]
    forbidden = [marker for marker in FORBIDDEN_MARKERS if marker in notebook_text]

    if missing:
        raise RuntimeError(f"Notebook is missing required models: {missing}")
    if forbidden:
        raise RuntimeError(f"Notebook still contains MLP markers: {forbidden}")


def main() -> int:
    validate_generator()
    runpy.run_path(str(GENERATOR_PATH), run_name="__main__")
    validate_notebook()
    print(f"Removed MLP and preserved LSTM/RNN/Transformer in: {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
