from __future__ import annotations

import argparse
import json
from pathlib import Path


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
        raise ValueError(f"Could not find {label} in target cell")
    return text.replace(old, new, 1)


def patch_cross_validation_cell(text: str) -> str:
    old_init = """    accepted_aug = False
    attempt_train_losses = []
    attempt_val_losses = []
    attempt_best_model_state = None
    attempt_best_val_prob = None
"""
    new_init = """    accepted_aug = False
    attempt_train_losses = []
    attempt_val_losses = []
    attempt_best_model_state = None
    attempt_best_val_prob = None
    model = None
    train_loader = None
    test_loader = None
    y_train_tensor = None
    y_test_tensor = None
    y_train_prob = None
    y_test_prob = None
    best_model_state = None
    best_val_prob = None
    train_losses = []
    val_losses = []
"""
    text = replace_once(text, old_init, new_init, "cross-validation variable initialization")

    old_after_attempts = """    # After attempts: use the last attempt's model / predictions
    if attempt_best_model_state is not None:
        # reload final chosen model state
        model.load_state_dict(attempt_best_model_state)
    
    # ensure y_train_prob and y_test_prob exist (from last attempt)
    try:
        y_train_prob
    except NameError:
        y_train_prob = get_predictions(model, train_loader, DEVICE)
    try:
        y_test_prob
    except NameError:
        y_test_prob = get_predictions(model, test_loader, DEVICE)

    # Sau vòng lặp, nạp lại trọng số tốt nhất
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Get predictions
    y_train_prob = get_predictions(model, train_loader, DEVICE)
    y_test_prob = get_predictions(model, test_loader, DEVICE)
"""
    new_after_attempts = """    # After attempts: use the final best model from this fold
    assert model is not None
    assert train_loader is not None
    assert test_loader is not None
    assert y_train_tensor is not None
    assert y_test_tensor is not None

    # Sau vòng lặp, nạp lại trọng số tốt nhất
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Get predictions
    y_train_prob = get_predictions(model, train_loader, DEVICE)
    y_test_prob = get_predictions(model, test_loader, DEVICE)
"""
    text = replace_once(text, old_after_attempts, new_after_attempts, "post-attempt prediction block")

    return text


def patch_notebook(notebook_path: Path) -> list[str]:
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    changed_cells: list[str] = []

    for index, cell in enumerate(data.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue

        text = cell_text(cell)
        original = text

        if "# Initialize cross-validation and storage" in text:
            text = patch_cross_validation_cell(text)

        if text != original:
            set_cell_source(cell, text)
            changed_cells.append(f"Cell {index}")

    if not changed_cells:
        raise RuntimeError("No matching notebook cells were updated. The notebook may already be patched or its structure changed.")

    notebook_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed_cells


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Bi-TCN notebook to satisfy Pylance unbound-variable checks.")
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
