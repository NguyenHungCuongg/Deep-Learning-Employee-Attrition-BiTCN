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


def patch_import_cell(text: str) -> str:
    if "import json" in text:
        return text
    old = """import warnings
"""
    new = """import warnings
import json
"""
    return replace_once(text, old, new, "json import")


def patch_model_cell(text: str) -> str:
    if "class AttentionLayer(nn.Module):" in text:
        return text

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


def patch_load_data_cell(text: str) -> str:
    if "ShapBiTCN" in text:
        return text

    start = text.find("# Paths")
    if start == -1:
        raise ValueError("Could not find load data cell in target cell")

    new_tail = """# Paths
DATA_PATH = Path("../data/processed/IBM_Cleaned.csv")
MODEL_PATH = Path("../results/models/bitcn_fold_1_best.pt")
MODEL_INFO_PATH = Path("../results/models/model_info.json")

# Load model metadata to recover the exact feature set used during training
with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
    model_info = json.load(f)
feature_names = model_info["feature_names"]

# Load data
df = pd.read_csv(DATA_PATH)
target_col = "Attrition"
id_cols = [c for c in df.columns if c.lower() in {"employee id", "employeeid", "employee_number"} or c.lower().endswith("id")]
X = df.drop(columns=[target_col] + id_cols, errors="ignore")
X = X.reindex(columns=feature_names, fill_value=0)
y = df[target_col].astype(int)

print(f"Data loaded: {X.shape}")
print(f"Checkpoint expects: {len(feature_names)} features")

# Local architecture copy to avoid stale kernel state from previous notebook runs
class ShapDilatedCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.5):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.padding > 0:
            x = torch.nn.functional.pad(x, (self.padding, 0))
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ShapBiTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.5):
        super().__init__()
        self.forward_conv = ShapDilatedCausalConv1d(in_channels, out_channels, kernel_size, dilation, dropout)
        self.backward_conv = ShapDilatedCausalConv1d(in_channels, out_channels, kernel_size, dilation, dropout)
        self.out_channels = out_channels * 2
        self.residual = nn.Identity() if in_channels == self.out_channels else nn.Conv1d(in_channels, self.out_channels, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        forward_out = self.forward_conv(x)
        backward_out = self.backward_conv(torch.flip(x, dims=[2]))
        backward_out = torch.flip(backward_out, dims=[2])
        bi_out = torch.cat([forward_out, backward_out], dim=1)
        bi_out = self.activation(bi_out + self.residual(x))
        return bi_out


class ShapBiTCNBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilations=(1, 2, 4), dropout=0.5):
        super().__init__()
        blocks = []
        current_channels = in_channels
        for dilation in dilations:
            blocks.append(ShapBiTCNBlock(current_channels, out_channels, kernel_size, dilation=dilation, dropout=dropout))
            current_channels = out_channels * 2
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ShapAttentionLayer(nn.Module):
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


class ShapBiTCN(nn.Module):
    def __init__(self, input_dim, num_features=None):
        super().__init__()
        if num_features is None:
            num_features = input_dim
        self.input_fc = nn.Linear(num_features, 128)
        self.input_bn = nn.BatchNorm1d(128)
        self.input_activation = nn.ReLU()
        self.branch1 = ShapBiTCNBranch(in_channels=1, out_channels=16, kernel_size=3, dilations=(1, 2, 4), dropout=0.5)
        self.branch2 = ShapBiTCNBranch(in_channels=1, out_channels=32, kernel_size=5, dilations=(1, 2, 4), dropout=0.5)
        self.attention = ShapAttentionLayer(channels=16 * 2 + 32 * 2, reduction=4, dropout=0.2)
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


# Initialize model and load weights
model = ShapBiTCN(input_dim=X.shape[1], num_features=X.shape[1])
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model loaded and set to eval mode")
"""

    return text[:start] + new_tail


def patch_notebook(notebook_path: Path) -> list[str]:
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    changed_cells: list[str] = []

    for index, cell in enumerate(data.get("cells", []), start=1):
        original = cell_text(cell)
        updated = original

        if cell.get("cell_type") == "code" and "import shap" in original and "import torch" in original:
            updated = patch_import_cell(updated)

        if cell.get("cell_type") == "code" and "class BiTCN(nn.Module):" in original:
            updated = patch_model_cell(updated)

        if cell.get("cell_type") == "code" and "# Paths" in original and "MODEL_PATH = Path(\"../results/models/bitcn_fold_1_best.pt\")" in original:
            updated = patch_load_data_cell(updated)

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
