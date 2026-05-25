"""
Utility script to ensure feature names are saved in config.json
This will update the config to include feature_names for SHAP analysis
"""

import json
from pathlib import Path
import pandas as pd

def update_config_with_features():
    """Update config.json to include feature names from IBM_Cleaned.csv"""
    
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "IBM_Cleaned.csv"
    CONFIG_PATH = PROJECT_ROOT / "outputs" / "bitcn_ibm" / "config.json"
    
    print("="*70)
    print("Updating config.json with feature names")
    print("="*70)
    
    # Load data to get feature names
    df = pd.read_csv(DATA_PATH)
    target_col = "Attrition"
    id_cols = [c for c in df.columns if c.lower() in {"employee id", "employeeid", "employee_number"} 
               or c.lower().endswith("id")]
    
    feature_names = df.drop(columns=[target_col] + id_cols, errors="ignore").columns.tolist()
    
    print(f"\nLoaded data from: {DATA_PATH}")
    print(f"Total features: {len(feature_names)}")
    print(f"Features: {feature_names[:5]}... (showing first 5)")
    
    # Load current config
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Check if feature_names already exists
    if "feature_names" in config:
        print(f"\nconfig.json already has feature_names: {len(config['feature_names'])} features")
    else:
        print(f"\nAdding feature_names to config.json...")
        config["feature_names"] = feature_names
        
        # Save updated config
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ config.json updated successfully")
        print(f"  - Added {len(feature_names)} feature names")
        print(f"  - Saved to: {CONFIG_PATH}")
    
    print("\n" + "="*70)
    print("SUCCESS! Config is now ready for SHAP analysis")
    print("="*70)
    
    return feature_names

if __name__ == "__main__":
    try:
        features = update_config_with_features()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
