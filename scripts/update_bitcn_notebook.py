from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

NOTEBOOK_DEFAULT = Path(__file__).resolve().parents[1] / "notebooks" / "3_BiTCN_Model_Training.ipynb"


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


def update_threshold_function(text: str) -> str:
    old = '''def find_optimal_threshold(y_true, y_prob, min_pos_preds=3):
    """
    Tìm ngưỡng phân loại tối ưu dựa trên F1-score cao nhất trên validation set.
    - Ưu tiên các ngưỡng tạo ít nhất `min_pos_preds` dự đoán dương để tránh trường hợp không có dự đoán lớp 1.
    - Nếu có nhiều ngưỡng cùng F1, chọn ngưỡng nhỏ hơn (độ nhạy cao hơn) để tránh trả về ngưỡng quá cao khiến test set không có dự đoán dương.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    thresholds = np.arange(0.01, 0.99, 0.01)
    candidates = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        n_pos = int(np.sum(y_pred))
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        f1 = f1_score(y_true, y_pred, zero_division=0)
        # Chỉ xem xét các ngưỡng tạo ít nhất min_pos_preds dự đoán dương và có ít nhất 1 true positive
        if n_pos >= min_pos_preds and tp > 0:
            candidates.append((t, f1))

    if len(candidates) > 0:
        # Chọn ngưỡng có F1 cao nhất; nếu tie, chọn ngưỡng nhỏ hơn
        candidates.sort(key=lambda x: (x[1], -x[0]), reverse=True)
        best_t, best_f1 = candidates[0]
        return float(best_t), float(best_f1)

    # Nếu không có ngưỡng nào thỏa điều kiện trên, fallback: chọn ngưỡng tối ưu theo F1 trên toàn miền,
    # ưu tiên ngưỡng nhỏ hơn khi bằng nhau (để tránh 0 predictions trên test)
    best_t = thresholds[0]
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if (f1 > best_f1) or (f1 == best_f1 and t < best_t):
            best_f1 = f1
            best_t = t

    return float(best_t), float(best_f1)
'''
    new = '''def find_optimal_threshold(y_true, y_prob, min_pos_preds=3, beta=2.0):
    """
    Tìm ngưỡng phân loại tối ưu dựa trên F-beta-score, mặc định là F2-score.
    - F2 đặt trọng số cho Recall cao hơn Precision để ưu tiên phát hiện attrition.
    - Ưu tiên các ngưỡng tạo ít nhất `min_pos_preds` dự đoán dương để tránh trường hợp không có dự đoán lớp 1.
    - Nếu có nhiều ngưỡng cùng điểm số, chọn ngưỡng nhỏ hơn (độ nhạy cao hơn).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    beta_sq = float(beta ** 2)
    thresholds = np.arange(0.01, 0.99, 0.01)
    candidates = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        n_pos = int(np.sum(y_pred))
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denominator = (beta_sq * precision) + recall
        f_beta = ((1 + beta_sq) * precision * recall / denominator) if denominator > 0 else 0.0

        # Chỉ xem xét các ngưỡng tạo ít nhất min_pos_preds dự đoán dương và có ít nhất 1 true positive
        if n_pos >= min_pos_preds and tp > 0:
            candidates.append((t, f_beta))

    if len(candidates) > 0:
        # Chọn ngưỡng có F2 cao nhất; nếu tie, chọn ngưỡng nhỏ hơn
        candidates.sort(key=lambda x: (x[1], -x[0]), reverse=True)
        best_t, best_score = candidates[0]
        return float(best_t), float(best_score)

    # Nếu không có ngưỡng nào thỏa điều kiện trên, fallback: chọn ngưỡng tối ưu theo F2 trên toàn miền,
    # ưu tiên ngưỡng nhỏ hơn khi bằng nhau (để tránh 0 predictions trên test)
    best_t = thresholds[0]
    best_score = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denominator = (beta_sq * precision) + recall
        f_beta = ((1 + beta_sq) * precision * recall / denominator) if denominator > 0 else 0.0

        if (f_beta > best_score) or (f_beta == best_score and t < best_t):
            best_score = f_beta
            best_t = t

    return float(best_t), float(best_score)
'''
    text = replace_once(text, old, new, "find_optimal_threshold")
    text = text.replace("F1-score", "F2-score")
    text = text.replace("F1", "F2")
    return text


def update_model_block(text: str) -> str:
    old = '''class BiTCNBranch(nn.Module):
    """Stack of bidirectional residual blocks for one ensemble path."""
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


class BiTCN(nn.Module):
    """Parallel ensemble Bi-TCN for binary attrition prediction."""
    def __init__(self, input_dim, num_features=None):
        super().__init__()
        if num_features is None:
            num_features = input_dim

        self.input_fc = nn.Linear(num_features, 128)
        self.input_bn = nn.BatchNorm1d(128)
        self.input_activation = nn.ReLU()

        # Reduced number of filters per user request: branch1 32->16, branch2 64->32
        self.branch1 = BiTCNBranch(in_channels=1, out_channels=16, kernel_size=3, dilations=(1, 2, 4), dropout=0.5)
        self.branch2 = BiTCNBranch(in_channels=1, out_channels=32, kernel_size=5, dilations=(1, 2, 4), dropout=0.5)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        # Adjust final FC input dims: 16*2 + 32*2
        self.fc_out = nn.Linear(16 * 2 + 32 * 2, 1)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.input_bn(x)
        x = self.input_activation(x)
        x = x.unsqueeze(1)

        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)

        x = torch.cat([branch1_out, branch2_out], dim=1)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x
'''
    new = '''class BiTCNBranch(nn.Module):
    """Stack of bidirectional residual blocks for one ensemble path."""
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
    """Channel attention layer to reweight Bi-TCN features before classification."""
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
    """Parallel ensemble Bi-TCN for binary attrition prediction."""
    def __init__(self, input_dim, num_features=None):
        super().__init__()
        if num_features is None:
            num_features = input_dim

        self.input_fc = nn.Linear(num_features, 128)
        self.input_bn = nn.BatchNorm1d(128)
        self.input_activation = nn.ReLU()

        # Reduced number of filters per user request: branch1 32->16, branch2 64->32
        self.branch1 = BiTCNBranch(in_channels=1, out_channels=16, kernel_size=3, dilations=(1, 2, 4), dropout=0.5)
        self.branch2 = BiTCNBranch(in_channels=1, out_channels=32, kernel_size=5, dilations=(1, 2, 4), dropout=0.5)

        self.attention = AttentionLayer(channels=16 * 2 + 32 * 2, reduction=4, dropout=0.2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        # Adjust final FC input dims: 16*2 + 32*2
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
'''
    return replace_once(text, old, new, "BiTCN block")


def update_hyperparameters(text: str) -> str:
    text = text.replace("L2_LAMBDA = 1e-4", "L2_LAMBDA = 1e-5")
    text = text.replace("OPTIMAL_THRESHOLD = 0.4 # Ngưỡng tối ưu sẽ được tính toán dựa trên F1-score", "OPTIMAL_THRESHOLD = 0.4 # Ngưỡng tối ưu sẽ được tính toán dựa trên F2-score")
    return text


def update_training_prints(text: str) -> str:
    text = text.replace("(F1-score:", "(F2-score:")
    text = text.replace("[OPTIMAL THRESHOLD] Found optimal threshold:", "[OPTIMAL THRESHOLD] Found optimal threshold:")
    return text


def patch_notebook(notebook_path: Path) -> list[str]:
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    changed_cells: list[str] = []

    for index, cell in enumerate(data.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue

        text = cell_text(cell)
        original = text

        if "def find_optimal_threshold(y_true, y_prob, min_pos_preds=3):" in text:
            text = update_threshold_function(text)

        if "L2_LAMBDA = 1e-4" in text or "OPTIMAL_THRESHOLD = 0.4 # Ngưỡng tối ưu sẽ được tính toán dựa trên F1-score" in text:
            text = update_hyperparameters(text)

        if "class BiTCNBranch(nn.Module):" in text and "class BiTCN(nn.Module):" in text:
            text = update_model_block(text)

        if "[OPTIMAL THRESHOLD] Found optimal threshold:" in text and "F1-score" in text:
            text = update_training_prints(text)

        if text != original:
            set_cell_source(cell, text)
            changed_cells.append(f"Cell {index}")

    if not changed_cells:
        raise RuntimeError("No matching notebook cells were updated. The notebook may already be patched or its structure changed.")

    notebook_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed_cells


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Bi-TCN notebook for higher recall.")
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
