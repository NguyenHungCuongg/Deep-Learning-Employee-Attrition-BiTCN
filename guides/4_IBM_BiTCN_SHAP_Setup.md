# 4_IBM_BiTCN_SHAP Notebook Setup Summary

## ✅ Completed Tasks

### 1. Created SHAP Analysis Notebook
- **File**: `notebooks/4_IBM_BiTCN_SHAP.ipynb`
- **Purpose**: Model explainability analysis using SHAP for the Bi-TCN model trained on IBM dataset
- **Structure**: 15 cells covering setup, model loading, SHAP calculation, and visualization

### 2. Created Setup Scripts
The following helper scripts were created in `/scripts`:

#### a. `generate_IBM_BiTCN_SHAP_notebook.py`
- Generates the main SHAP notebook from scratch
- Configurable cell contents for easy customization
- Usage: `python scripts/generate_IBM_BiTCN_SHAP_notebook.py`

#### b. `update_config_with_features.py`
- Updates `config.json` with feature names from data
- Ensures feature names are available for SHAP analysis
- Usage: `python scripts/update_config_with_features.py`

#### c. `validate_shap_notebook_setup.py`
- Validates all prerequisites for running the notebook
- Provides detailed setup guidance and troubleshooting tips
- Usage: `python scripts/validate_shap_notebook_setup.py`

### 3. Updated Configuration
- **File**: `outputs/bitcn_ibm/config.json`
- **Update**: Added `feature_names` (43 features from IBM dataset)
- This ensures the notebook can properly load and align features during SHAP analysis

## 📋 Notebook Structure

The `4_IBM_BiTCN_SHAP.ipynb` notebook contains:

| Cell | Type | Purpose |
|------|------|---------|
| 1 | Markdown | Title and introduction |
| 2 | Code | Setup imports and device configuration |
| 3 | Markdown | Architecture definition section header |
| 4 | Code | BiTCN, DilatedCausalConv1d, BiTCNBlock, BiTCNBranch, AttentionLayer classes |
| 5 | Markdown | Load Model & Data section header |
| 6 | Code | Load data from `data/processed/IBM_Cleaned.csv`, load model from `outputs/bitcn_ibm/models/best_model_fold_1.pth` |
| 7 | Markdown | Preprocessing for SHAP section header |
| 8 | Code | SHAP wrapper function and explainer initialization |
| 9 | Markdown | Calculate SHAP Values section header |
| 10 | Code | Calculate SHAP values for 200 sample subset |
| 11 | Markdown | Visualizations section header |
| 12 | Code | Feature importance bar plot |
| 13 | Code | Beeswarm plot showing directional impact |
| 14 | Markdown | Interpretation guide with insights |
| 15 | Code | Extract top features and save as CSV |

## 🚀 Quick Start

1. **Open Notebook**:
   ```
   notebooks/4_IBM_BiTCN_SHAP.ipynb
   ```

2. **Run Cells Sequentially**:
   - Cells 1-8: Setup and model initialization (~1 minute)
   - Cell 9-10: SHAP value calculation (~2-5 minutes)
   - Cells 11-15: Visualizations and analysis (~30 seconds)

3. **Expected Outputs**:
   - PNG plots in `outputs/bitcn_ibm/figures/`:
     - `shap_feature_importance_bar.png`
     - `shap_beeswarm_plot.png`
   - CSV file: `outputs/bitcn_ibm/shap_feature_importance.csv`

## 📊 Key Features

### Model Loading
- Loads the best Bi-TCN model from `outputs/bitcn_ibm/models/best_model_fold_1.pth`
- Model architecture matches exactly with `3_BiTCN_IBM_Training.ipynb`
- Supports loading other models (GAN-augmented, SMOTE, ADASYN, etc.)

### SHAP Analysis
- **Background Data**: 100 samples for explainer initialization
- **Analysis Sample**: 200 samples for detailed SHAP computation
- **Top Features**: Displays 15 most important features by default

### Visualization Types
- **Bar Plot**: Shows mean |SHAP| value per feature (global importance)
- **Beeswarm Plot**: Shows individual impact of feature values (directional effect)

## 🎯 Important Notes

### Data Alignment
- All 43 features are automatically extracted from `IBM_Cleaned.csv`
- Feature order is preserved and validated
- No manual feature selection needed

### Model Path
- Uses `best_model_fold_1.pth` (model trained without augmentation, RAW)
- Alternative models available:
  - `gan_best_model_fold_1.pth` (GAN-augmented)
  - `smote_best_model_fold_1.pth` (SMOTE-augmented)
  - `adasyn_best_model_fold_1.pth` (ADASYN-augmented)
  - `randomoversampling_best_model_fold_1.pth` (RandomOverSampling-augmented)

### Customization
To analyze different models or parameters:

```python
# Cell 6 - Change model path
MODEL_PATH = MODEL_DIR / "gan_best_model_fold_1.pth"  # For GAN model

# Cell 8 - Adjust background data size
background_data = shap.sample(X, 50)  # Smaller = faster

# Cell 10 - Adjust analysis sample size
test_subset = X.sample(100, random_state=42)  # Smaller = faster

# Cells 12-13 - Adjust displayed features
shap.summary_plot(shap_values, test_subset, plot_type="bar", max_display=20)
```

## 📝 Configuration Details

### Feature Names (43 total)
Age, BusinessTravel, DailyRate, DistanceFromHome, Education, EducationField, EmployeeCount, EmployeeNumber, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager, EmployeeID (removed for analysis)

### Data Statistics
- **Total Samples**: 1,470
- **Class Distribution**: 
  - No Attrition (0): 1,233 (83.88%)
  - Attrition (1): 237 (16.12%)
- **Features Used**: 43 (after ID removal)

## 🔧 Troubleshooting

### Common Issues

1. **Model not found**
   - Check: `outputs/bitcn_ibm/models/best_model_fold_1.pth` exists
   - Solution: Run `3_BiTCN_IBM_Training.ipynb` first or verify file paths

2. **Feature name mismatch**
   - Check: `outputs/bitcn_ibm/config.json` has 43 feature_names
   - Solution: Run `scripts/update_config_with_features.py`

3. **SHAP calculation slow**
   - Reduce background_data size (Cell 8): from 100 to 50
   - Reduce test_subset size (Cell 10): from 200 to 100
   - Use GPU if available (modify Cell 2 to use CUDA)

4. **Out of memory**
   - Reduce both background_data and test_subset sizes
   - Consider running on a machine with more RAM
   - Process in batches (advanced usage)

## ✨ Interpretation Guide

### Understanding SHAP Plots

**Bar Plot (Feature Importance)**:
- Shows average |SHAP| value for each feature
- Larger bars = more important features
- Typically: OverTime, MonthlyIncome, JobLevel, StockOptionLevel are top predictors

**Beeswarm Plot (Directional Impact)**:
- Each dot = one sample's feature value and its SHAP contribution
- Red dots on right = higher feature value increases attrition likelihood
- Red dots on left = higher feature value decreases attrition likelihood
- Color gradient shows feature value magnitude

**Expected Key Features**:
- **OverTime**: Strong positive correlation with attrition
- **MonthlyIncome**: Negative correlation (higher income = lower attrition)
- **StockOptionLevel**: Negative correlation (benefits reduce attrition)
- **JobSatisfaction**: Likely negative correlation

## 📚 Related Files

- **Training notebook**: `notebooks/3_BiTCN_IBM_Training.ipynb`
- **Baseline comparison**: `notebooks/2_Baselines_Augmentation.ipynb`
- **EDA and preprocessing**: `notebooks/1_EDA_Preprocessing.ipynb`
- **Original SHAP notebook**: `notebooks/4_XAI_Evaluation_SHAP.ipynb` (for 3_BiTCN_Model_Training.ipynb)

## 📞 Support Scripts

To regenerate or validate the notebook:

```bash
# Validate all prerequisites
python scripts/validate_shap_notebook_setup.py

# Update config with features
python scripts/update_config_with_features.py

# Regenerate notebook from scratch
python scripts/generate_IBM_BiTCN_SHAP_notebook.py
```

---

**Created**: 2024
**Model**: Bi-TCN with Attention Layer
**Dataset**: IBM HR Analytics Employee Attrition (1,470 samples, 43 features)
