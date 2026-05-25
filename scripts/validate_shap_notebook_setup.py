"""
Complete setup and validation script for 4_IBM_BiTCN_SHAP.ipynb
This script verifies all prerequisites and provides guidance
"""

import json
from pathlib import Path
import pandas as pd

def check_prerequisites():
    """Check if all required files and configurations exist"""
    
    PROJECT_ROOT = Path(__file__).parent.parent
    
    checks = {
        "Data file": PROJECT_ROOT / "data" / "processed" / "IBM_Cleaned.csv",
        "Model checkpoint (Fold 1)": PROJECT_ROOT / "outputs" / "bitcn_ibm" / "models" / "best_model_fold_1.pth",
        "Config file": PROJECT_ROOT / "outputs" / "bitcn_ibm" / "config.json",
        "Notebook": PROJECT_ROOT / "notebooks" / "4_IBM_BiTCN_SHAP.ipynb",
        "Figures directory": PROJECT_ROOT / "outputs" / "bitcn_ibm" / "figures",
    }
    
    print("="*80)
    print("PREREQUISITES CHECK FOR 4_IBM_BiTCN_SHAP.ipynb")
    print("="*80)
    
    all_ok = True
    for name, path in checks.items():
        exists = path.exists()
        status = "✓ OK" if exists else "✗ MISSING"
        print(f"{name:35s}: {status}")
        if not exists:
            all_ok = False
            print(f"  Expected at: {path}")
    
    # Check config has feature_names
    config_path = PROJECT_ROOT / "outputs" / "bitcn_ibm" / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        has_features = "feature_names" in config
        status = "✓ OK" if has_features else "✗ MISSING"
        print(f"\nconfig.json has feature_names      : {status}")
        if has_features:
            print(f"  - {len(config['feature_names'])} features found")
        else:
            all_ok = False
    
    # Check data shape
    data_path = PROJECT_ROOT / "data" / "processed" / "IBM_Cleaned.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"\nData shape: {df.shape[0]} samples × {df.shape[1]} features")
        print(f"Attrition distribution:")
        print(df["Attrition"].value_counts().to_string())
    
    return all_ok

def provide_guidance():
    """Provide step-by-step guidance"""
    
    print("\n" + "="*80)
    print("SETUP COMPLETE - NEXT STEPS")
    print("="*80)
    
    steps = """
1. OPEN NOTEBOOK:
   - Navigate to: notebooks/4_IBM_BiTCN_SHAP.ipynb
   - Open in VS Code Jupyter extension

2. NOTEBOOK STRUCTURE:
   - Cell 1-2: Setup and imports
   - Cell 3-4: Define BiTCN architecture (same as training model)
   - Cell 5-6: Load model and data from outputs/bitcn_ibm
   - Cell 7-8: SHAP wrapper function and explainer initialization
   - Cell 9-10: Calculate SHAP values (200 samples)
   - Cell 11-13: Generate visualizations (bar plot, beeswarm plot)
   - Cell 14: Interpretation guide
   - Cell 15: Extract top features and save results

3. RUN CELLS SEQUENTIALLY:
   - Ensure virtual environment is activated
   - Run cells 1-8 first for setup
   - Run cell 9 to calculate SHAP values (takes ~2-5 minutes)
   - Run cells 10-15 for visualizations and analysis

4. EXPECTED OUTPUTS:
   - SHAP summary plots (PNG files in outputs/bitcn_ibm/figures/)
   - Feature importance rankings (CSV file)
   - Console output with top features

5. INTERPRETATION:
   - Bar plot: Shows average |SHAP| value per feature
   - Beeswarm plot: Shows individual sample impacts by feature value
   - CSV output: Detailed feature importance rankings

6. CUSTOMIZATION OPTIONS:
   - Modify background_data size (line ~165): larger = more accurate, slower
   - Modify test_subset size (line ~178): larger = more comprehensive, slower
   - Change max_display in plots (line ~210, 216): adjust top features shown

7. TROUBLESHOOTING:
   - If model loading fails: Check outputs/bitcn_ibm/models/best_model_fold_1.pth exists
   - If SHAP calculation is slow: Reduce background_data and test_subset sizes
   - If feature names mismatch: config.json feature_names must match data columns
   - Available models: best_model_fold_*.pth (RAW), gan_best_model_fold_*.pth (GAN-augmented),
     smote_best_model_fold_*.pth (SMOTE), adasyn_best_model_fold_*.pth (ADASYN),
     randomoversampling_best_model_fold_*.pth (RandomOverSampling)

IMPORTANT:
   - This notebook uses the BEST model (Fold 1) from 3_BiTCN_IBM_Training.ipynb
   - To analyze other folds, modify MODEL_PATH in cell 6
   - SHAP values are computed on CPU by default (add GPU support if needed)
"""
    
    print(steps)

def main():
    print("\n")
    
    # Check prerequisites
    all_ok = check_prerequisites()
    
    if all_ok:
        print("\n✓ All prerequisites are met!")
    else:
        print("\n✗ Some prerequisites are missing. Please check the above.")
    
    # Provide guidance
    provide_guidance()
    
    print("\n" + "="*80)
    print("READY TO USE 4_IBM_BiTCN_SHAP.ipynb")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
