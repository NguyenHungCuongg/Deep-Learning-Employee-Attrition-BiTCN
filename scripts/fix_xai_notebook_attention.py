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
        raise ValueError(f"Could not find {label} in target cell")
    return text.replace(old, new, 1)


def patch_model_cell(text: str) -> str:
    new_block = """class BiTCNBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilations=(1, 2, 4), dropout=0.5):
        super().__init__()
        blocks = []
        current_channels = in_channels
        for dilation in dilations:
            blocks.append(BiTCNBlock(current_channels, out_channels, kernel_size, dilation=dilation, dropout=dropout))
            current_channels = out_channels * 2
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

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

class BiTCN(nn.Module):
    def __init__(self, input_dim, num_features=None):
        super().__init__()
        if num_features is None:
            num_features = input_dim
        self.input_fc = nn.Linear(num_features, 128)
        self.input_bn = nn.BatchNorm1d(128)
        self.input_activation = nn.ReLU()
        self.branch1 = BiTCNBranch(in_channels=1, out_channels=16, kernel_size=3, dilations=(1, 2, 4), dropout=0.5)
        self.branch2 = BiTCNBranch(in_channels=1, out_channels=32, kernel_size=5, dilations=(1, 2, 4), dropout=0.5)
        self.attention = AttentionLayer(channels=16 * 2 + 32 * 2, reduction=4, dropout=0.2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(16 * 2 + 32 * 2, 1)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.input_bn(x)
        x = self.input_activation(x)
        x = x.unsqueeze(1)
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        x = torch.cat([branch1_out, branch2_out], dim=1)
        x = self.attention(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x
"""
    start = text.find("class BiTCNBranch(nn.Module):")
    if start == -1:
        raise ValueError("Could not find BiTCN architecture block in target cell")

    return text[:start] + new_block


def patch_notebook(notebook_path: Path) -> list[str]:
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    changed_cells: list[str] = []

    for index, cell in enumerate(data.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue

        text = cell_text(cell)
        original = text

        if "class BiTCNBranch(nn.Module):" in text and "class BiTCN(nn.Module):" in text:
            text = patch_model_cell(text)

        if text != original:
            set_cell_source(cell, text)
            changed_cells.append(f"Cell {index}")

    if not changed_cells:
        raise RuntimeError("No matching notebook cells were updated. The notebook may already be patched or its structure changed.")

    notebook_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed_cells


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch the XAI notebook so its BiTCN architecture matches the trained checkpoint.")
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
