from __future__ import annotations

import argparse
import json
from pathlib import Path


NOTEBOOK_DEFAULT = Path(__file__).resolve().parents[1] / "notebooks" / "4_XAI_Evaluation_SHAP.ipynb"


def cell_text(cell: dict) -> str:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    return "".join(source)


def set_cell_source(cell: dict, new_text: str) -> None:
    cell["source"] = new_text.splitlines(keepends=True)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise ValueError(f"Could not find '{label}' in target cell")
    return text.replace(old, new, 1)


def patch_model_cell(text: str) -> str:
    if "class AttentionLayer(nn.Module):" not in text:
        old = """return x

class BiTCN(nn.Module):"""
        new = """return x

class AttentionLayer(nn.Module):
    def __init__(self, channels, reduction=4, dropout=0.2):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.pool(x).flatten(1)
        weights = self.gate(weights).unsqueeze(-1)
        return x * weights

class BiTCN(nn.Module):"""
        text = replace_once(text, old, new, "AttentionLayer insertion")

    if "self.attention = AttentionLayer(" not in text:
        old = """        self.branch1 = BiTCNBranch(in_channels=1, out_channels=16, kernel_size=3, dilations=(1, 2, 4), dropout=0.5)
        self.branch2 = BiTCNBranch(in_channels=1, out_channels=32, kernel_size=5, dilations=(1, 2, 4), dropout=0.5)
        self.pool = nn.AdaptiveAvgPool1d(1)
"""
        new = """        self.branch1 = BiTCNBranch(in_channels=1, out_channels=16, kernel_size=3, dilations=(1, 2, 4), dropout=0.5)
        self.branch2 = BiTCNBranch(in_channels=1, out_channels=32, kernel_size=5, dilations=(1, 2, 4), dropout=0.5)
        self.attention = AttentionLayer(channels=16 * 2 + 32 * 2, reduction=4, dropout=0.2)
        self.pool = nn.AdaptiveAvgPool1d(1)
"""
        text = replace_once(text, old, new, "AttentionLayer field")

    if "x = self.attention(x)" not in text:
        old = """        x = torch.cat([branch1_out, branch2_out], dim=1)
        x = self.pool(x).flatten(1)
"""
        new = """        x = torch.cat([branch1_out, branch2_out], dim=1)
        x = self.attention(x)
        x = self.pool(x).flatten(1)
"""
        text = replace_once(text, old, new, "Attention forward pass")

    return text


def patch_intro_markdown(text: str) -> str:
    text = text.replace(
        "Notebook này cung cấp model explainability cho Bi-TCN attrition prediction model sử dụng SHAP (SHapley Additive exPlanations).",
        "Notebook này cung cấp model explainability cho Bi-TCN attrition prediction model đã được bổ sung Attention Layer, sử dụng SHAP (SHapley Additive exPlanations).",
    )
    text = text.replace(
        "   1. Load trained Bi-TCN model và preprocessed data.",
        "   1. Load trained Bi-TCN model có Attention và preprocessed data.",
    )
    return text


def patch_notebook(notebook_path: Path) -> list[str]:
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    changed_cells: list[str] = []

    for index, cell in enumerate(data.get("cells", []), start=1):
        original = cell_text(cell)
        updated = original

        if cell.get("cell_type") == "code" and "class BiTCN(nn.Module):" in original:
            updated = patch_model_cell(updated)

        if cell.get("cell_type") == "markdown" and "Notebook này cung cấp model explainability" in original:
            updated = patch_intro_markdown(updated)

        if updated != original:
            set_cell_source(cell, updated)
            changed_cells.append(f"Cell {index}")

    if not changed_cells:
        raise RuntimeError("No matching notebook cells were updated. The notebook may already be patched or its structure changed.")

    notebook_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed_cells


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch XAI notebook for the Attention-based Bi-TCN model.")
    parser.add_argument("notebook", nargs="?", default=str(NOTEBOOK_DEFAULT), help="Path to the notebook to patch")
    args = parser.parse_args()

    notebook_path = Path(args.notebook)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    changed_cells = patch_notebook(notebook_path)
    print(f"Updated {notebook_path}")
    print("Changed:")
    for item in changed_cells:
        print(f"- {item}")


if __name__ == "__main__":
    main()
