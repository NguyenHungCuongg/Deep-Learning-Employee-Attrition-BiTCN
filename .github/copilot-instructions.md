# Bi-TCN Employee Attrition Research Project Instructions

## 1. Primary Research Goal

This project aims to:

1. Faithfully reproduce the Bi-TCN employee attrition paper:
   "A Deep Learning Model Based on Bidirectional Temporal Convolutional Network (Bi-TCN) for Predicting Employee Attrition"

2. Improve the original architecture using modern deep learning techniques.

3. Conduct rigorous experiments on:
   - IBM HR Analytics Employee Attrition
   - Kaggle Employee Churn Dataset

4. Benchmark against:
   - Classical ML models
   - Deep Learning models
   - Multiple Data Augmentation strategies

5. Produce reproducible, publication-quality experimental results.

---

# 2. Mandatory Research Standards

## Reproducibility

Always:

- Set random seeds:
  - numpy
  - torch
  - random
- Enable deterministic training where possible.
- Save:
  - configs
  - metrics
  - checkpoints
  - logs
  - confusion matrices
  - ROC curves

Use:

- TensorBoard or Weights & Biases logging.

---

## Cross Validation

Use:

- Stratified 5-Fold Cross Validation

Rules:

- Fit preprocessing ONLY on training fold.
- Apply transform to validation fold.
- Never allow data leakage.

For augmentation:

- Apply augmentation ONLY to training fold.
- Never augment validation/test data.

---

# 3. Dataset Handling

## IBM Dataset

Characteristics:

- Small dataset (~1470 samples)
- Imbalanced classes

Requirements:

- Strong regularization
- Extensive augmentation experiments
- Smaller model capacity

Recommended:

- 16/32 filters instead of 32/64
- Dropout >= 0.4
- Weight decay
- Early stopping

---

## Kaggle Employee Churn Dataset

Characteristics:

- Larger dataset (~14k samples)
- Lower imbalance

Requirements:

- Compare raw vs augmented training
- Avoid unnecessary oversampling

---

# 4. Data Preprocessing Pipeline

Always implement:

## Cleaning

- Remove duplicates
- Handle missing values
- Validate label consistency

## Encoding

Use:

- OneHotEncoder for nominal categorical features
  OR
- Learned embeddings for deep learning models

Avoid arbitrary ordinal encoding unless ordinal meaning exists.

---

## Feature Scaling

Use:

- MinMaxScaler for deep learning models
- StandardScaler for ML baselines where appropriate

Scaler MUST be fit only on training fold.

---

## Feature Engineering

Create optional engineered features:

- Income per working year
- Promotion frequency
- Satisfaction-to-tenure ratio
- Overtime intensity indicators

Evaluate whether engineered features improve performance.

---

# 5. Data Augmentation Experiments

Implement and compare:

## Traditional

- Random Oversampling
- SMOTE
- Borderline-SMOTE
- ADASYN

## Generative

- GAN
- CTGAN (preferred for tabular data)

Rules:

- Augmentation only on training fold
- Compare augmentation effects statistically

For IBM:

- Strong augmentation encouraged

For Kaggle:

- Prefer minimal augmentation unless proven beneficial

---

# 6. Bi-TCN Architecture Requirements

## Core Architecture

Must implement:

- Bidirectional TCN branches
- Dilated causal convolutions
- Residual connections
- Forward/backward sequence processing

---

## Original Paper Settings

Bi-TCN Block 1:

- kernel = 3
- dilation = 1

Bi-TCN Block 2:

- kernel = 5
- dilation = 2

---

## Mandatory Improvements

### Attention Mechanism

Implement at least one:

- SE Block
- CBAM
- Self-Attention
- Multi-Head Attention

Preferred:

- Lightweight channel attention

---

### Global Pooling

Use:

- Global Average Pooling

instead of large flatten layers.

---

### Residual Stabilization

Use:

- Residual skip connections
- BatchNorm or LayerNorm

---

### Overfitting Prevention

Mandatory:

- Dropout
- Weight decay
- Early stopping

Recommended:

- Label smoothing
- Gaussian noise
- Mixout

---

# 7. Training Strategy

## Optimizer

Default:

- AdamW

Recommended:

- learning_rate = 1e-3
- weight_decay = 1e-4

---

## Scheduler

Use one:

- ReduceLROnPlateau
- CosineAnnealingLR
- OneCycleLR

---

## Loss Function

Compare:

- Binary Cross Entropy
- Focal Loss

Use Focal Loss for imbalanced settings.

---

## Threshold Optimization

Do NOT use fixed 0.5 threshold blindly.

Optimize threshold using:

- F1-score
- F2-score
- Youden Index

Store best threshold per fold.

---

# 8. Baseline Benchmark Models

## Classical ML

Implement:

- Logistic Regression
- Random Forest
- XGBoost
- CatBoost
- LightGBM

Use:

- Optuna or Bayesian optimization

for fair hyperparameter tuning.

---

## Deep Learning Baselines

Implement:

- MLP
- CNN
- LSTM
- Bi-LSTM
- GRU
- Transformer

Optional:

- TabNet
- FT-Transformer
- SAINT

---

# 9. Evaluation Requirements

Track:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Especially prioritize:

- Recall
- F1
- PR-AUC

due to class imbalance.

---

## Visualization

Generate:

- ROC curves
- PR curves
- Confusion matrices
- Loss curves
- Calibration curves

---

## Statistical Validation

Perform:

- Mean ± std across folds
- Paired t-test or Wilcoxon test

to validate improvement significance.

---

# 10. Explainable AI (XAI)

Implement:

- SHAP

Analyze:

- Global feature importance
- Local explanations

Focus on:

- OverTime
- JobLevel
- MonthlyIncome
- StockOptionLevel
- JobSatisfaction

---

# 11. Ablation Study (Mandatory)

Run experiments removing:

- Attention layer
- GAN augmentation
- Residual connections
- Scheduler
- Feature engineering

Demonstrate contribution of each component.

---

# 12. Expected Final Deliverables

The project must produce:

- Reproducible codebase
- Final trained models
- Experimental tables
- Comparison charts
- Ablation study
- SHAP explanations
- Publication-quality figures
- Final research report

---

# 13. Preferred Tech Stack

Frameworks:

- PyTorch
- scikit-learn
- imbalanced-learn
- XGBoost
- LightGBM
- CatBoost

Visualization:

- matplotlib
- seaborn
- plotly

Tracking:

- TensorBoard or WandB

Hyperparameter tuning:

- Optuna

Explainability:

- SHAP

---

# 14. Important Constraints

Never:

- Use test data during preprocessing fitting
- Oversample before train/test split
- Compare models with unfair tuning budgets
- Report single-run results only

Always:

- Average across folds
- Save best checkpoints
- Report reproducible metrics
- Use statistically meaningful comparisons
