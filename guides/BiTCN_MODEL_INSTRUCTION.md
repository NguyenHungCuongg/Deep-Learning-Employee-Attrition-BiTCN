# BiTCN_MODEL_INSTRUCTION.md

# Objective

Generate a Python training pipeline that creates:

```bash
3_BiTCN_IBM_Training.ipynb
```

The notebook must reproduce and improve the Bi-TCN employee attrition model from the research paper:

> "A Deep Learning Model Based on Bidirectional Temporal Convolutional Network (Bi-TCN) for Predicting Employee Attrition"

The notebook must train on:

```bash
IBM_Cleaned.csv
```

which has already been preprocessed in:

```bash
1_EDA_Preprocessing.ipynb
```

and compared against baseline models implemented in:

```bash
2_Baselines.ipynb
```

---

# General Rules

The generated notebook must:

- Be clean and modular.
- Be reproducible.
- Be suitable for academic/research experiments.
- Avoid data leakage.
- Prevent severe overfitting.
- Produce publication-quality outputs.

The notebook should be generated from a `.py` script if necessary.

Example:

```bash
generate_BiTCN_notebook.py
```

that exports:

```bash
3_BiTCN_IBM_Training.ipynb
```

Use:

- nbformat
- jupyter notebook JSON generation
- or jupytext

---

# Notebook Structure

The notebook MUST contain the following sections in order.

---

# 1. Project Setup

Include:

- imports
- device detection
- reproducibility setup
- warnings suppression

Set seeds for:

- random
- numpy
- torch

Enable deterministic behavior.

Example:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

# 2. Load Dataset

Load:

```bash
IBM_Cleaned.csv
```

Requirements:

- validate missing values
- print dataset shape
- print class distribution
- verify target column

Expected target:

```python
Attrition
```

---

# 3. Feature / Target Split

Split into:

```python
X
y
```

Requirements:

- convert to numpy
- convert to float32
- reshape for TCN input

Expected TCN shape:

```python
(batch, channels, sequence_length)
```

For tabular data:

- use feature dimension as sequence length
- use 1 channel

Example:

```python
X = X.reshape(samples, 1, num_features)
```

---

# 4. Cross Validation Pipeline

Use:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Rules:

- preprocessing fit ONLY on train fold
- validation fold must never leak
- augmentation ONLY on training fold

---

# 5. Data Augmentation

Implement configurable augmentation pipeline.

Support:

- RAW
- RandomOversampling
- SMOTE
- ADASYN

Optional:

- CTGAN
- GAN

Default:

- SMOTE

Use:

```python
imbalanced-learn
```

Requirements:

- augmentation ONLY on train fold
- print class distribution after augmentation

---

# 6. Tensor Dataset & DataLoader

Create:

- TensorDataset
- DataLoader

Requirements:

- pin_memory=True
- num_workers configurable
- shuffle=True for train only

---

# 7. Bi-TCN Architecture

Implement a modular PyTorch model.

Required classes:

- Chomp1d
- TemporalBlock
- TemporalConvNet
- ChannelAttention
- BiTCNClassifier

---

# 8. Original Paper Requirements

Reproduce paper architecture:

Bi-TCN Branch 1:

- kernel_size = 3
- dilation = 1

Bi-TCN Branch 2:

- kernel_size = 5
- dilation = 2

Bidirectional processing:

- forward branch
- backward branch using torch.flip

Use:

- residual connections
- causal convolutions

---

# 9. Required Improvements

Mandatory improvements over paper:

## Channel Attention

Implement SE-like channel attention.

Requirements:

- lightweight
- adaptive channel weighting

Example:

- squeeze-excitation
- adaptive average pooling
- FC reduction

---

## Global Average Pooling

Use:

```python
AdaptiveAvgPool1d(1)
```

Avoid large flatten layers.

---

## Regularization

Must include:

- Dropout >= 0.4
- Weight decay
- Early stopping

---

## Normalization

Use one:

- BatchNorm1d
- LayerNorm

---

# 10. Model Configuration

Recommended default config:

```python
FILTERS_1 = 16
FILTERS_2 = 32

DROPOUT = 0.5

LR = 1e-3
WEIGHT_DECAY = 1e-4

BATCH_SIZE = 64
EPOCHS = 50
```

IBM dataset is small.

Avoid overly large models.

---

# 11. Loss Function

Support:

- BCEWithLogitsLoss
- FocalLoss

Default:

```python
BCEWithLogitsLoss(pos_weight=...)
```

Automatically compute:

```python
pos_weight
```

from training fold.

---

# 12. Optimizer & Scheduler

Use:

```python
AdamW
```

Use scheduler:

```python
ReduceLROnPlateau
```

Monitor:

```python
validation_f1
```

---

# 13. Training Loop

The training loop must:

- support GPU
- support mixed precision if CUDA available
- track all metrics
- save best fold checkpoint

Track:

- train loss
- validation loss
- accuracy
- precision
- recall
- F1
- ROC-AUC

Use:

```python
tqdm
```

for progress bars.

---

# 14. Threshold Optimization

Do NOT use fixed threshold 0.5 blindly.

Implement:

```python
find_best_threshold()
```

Optimize threshold using:

- F1-score
- PR curve

Store best threshold per fold.

---

# 15. Early Stopping

Implement:

```python
patience = 10
```

Monitor:

```python
validation_f1
```

Restore best weights.

---

# 16. Evaluation

Generate for each fold:

- confusion matrix
- ROC curve
- PR curve
- classification report

Generate final:

- mean ± std metrics

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

---

# 17. Visualization

Generate:

- train vs validation loss
- train vs validation F1
- ROC curves
- confusion matrices

Use:

- matplotlib
- seaborn

Figures must be publication-quality.

---

# 18. Explainable AI (Optional but Preferred)

Implement SHAP:

- DeepExplainer
  OR
- KernelExplainer

Visualize:

- feature importance
- top influential features

Focus on:

- OverTime
- MonthlyIncome
- JobLevel
- StockOptionLevel

---

# 19. Compare Against Baselines

Load baseline results from:

```bash
2_Baselines.ipynb
```

OR:

```bash
baseline_results.csv
```

Create comparison tables:

- RF vs XGBoost vs MLP vs CNN vs Bi-TCN

Highlight:

- best F1
- best Recall
- best ROC-AUC

---

# 20. Output Files

Save:

- best_model_fold_X.pth
- metrics.csv
- fold_results.csv
- roc_curve.png
- confusion_matrix.png

Create:

```bash
outputs/
models/
figures/
logs/
```

automatically.

---

# 21. Coding Standards

Requirements:

- modular code
- reusable functions
- clear comments
- no duplicated logic

Avoid:

- giant notebook cells
- hardcoded paths
- magic numbers

Use:

- config dictionary
  OR
- dataclass configuration

---

# 22. Important Constraints

NEVER:

- fit scaler before cross-validation
- oversample validation data
- leak validation data into training
- report single-run metrics only

ALWAYS:

- average across folds
- save best checkpoints
- use reproducible seeds
- report mean ± std

---

# 23. Preferred Libraries

Use:

- PyTorch
- scikit-learn
- imbalanced-learn
- pandas
- numpy
- matplotlib
- seaborn
- tqdm

Optional:

- SHAP
- Optuna

---

# 24. Final Goal

The generated notebook should:

1. Faithfully reproduce the original Bi-TCN paper.
2. Improve generalization performance.
3. Reduce overfitting on IBM dataset.
4. Achieve strong Recall and F1-score.
5. Produce research-quality experimental outputs.
6. Be extendable later to Kaggle Employee Churn dataset.
