# Context: Fixing Imbalanced Class Prediction Issue in Fold 2

## 1. Problem Description
In the current implementation of `3_BiTCN_Model_Training_ver2.ipynb`, **Fold 2** shows a performance anomaly:
- **Accuracy**: 0.8367 (matches majority class proportion).
- **AUC**: 0.8122 (highest among all folds, indicating good separation ability).
- **Precision, Recall, F1**: All are **0.0000**.

## 2. Technical Root Cause
The model is effectively learning features (as shown by the high AUC), but it is biased towards the majority class (No Attrition). Because the default classification threshold is set to **0.5**, and the minority class probability outputs never cross this threshold in Fold 2, the model predicts "0" for every single instance.

## 3. Required Fixes (Strategic Improvements)
To resolve this, we need to implement three specific technical changes in the code:

### A. Class Weighting in Loss Function
Update the loss function to penalize errors on the "Attrition" (Yes) class more heavily.
- **Change**: Pass `pos_weight` to `nn.BCEWithLogitsLoss()`.
- **Rationale**: The ratio of No:Yes is roughly 5.2:1 (1233 vs 237).

### B. Increase Early Stopping Patience
Fold 2 stopped very early (Epoch 16) while Val Loss was still at its lowest point earlier (Epoch 9).
- **Change**: Increase `PATIENCE` from 7 to **15**.
- **Rationale**: Allow the model more time to "fine-tune" on the hard-to-learn minority samples before stopping.

### C. Dynamic Threshold Selection
Instead of a fixed 0.5 threshold, the evaluation logic should identify the threshold that maximizes the F1-score for each fold.
- **Change**: Update `compute_metrics` or the evaluation loop to find the best threshold based on the validation set.
