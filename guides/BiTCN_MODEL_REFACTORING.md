# Context for Bi-TCN Model Refactoring

## 1. Objective
Refactor the existing `BiTCN` class in PyTorch to strictly follow the architecture proposed in the research paper: **"A Deep Learning Model Based on Bidirectional Temporal Convolutional Network (Bi-TCN) for Predicting Employee Attrition"** (Source: `applsci-15-02984.pdf` or `PAPER_SUMMARY.md`).

## 2. Theoretical Architecture (from Figure 2 & Table 4)
The current implementation is a sequential/stacked version. It must be converted into a **Parallel Ensemble** structure as described in the paper:

1.  **Input Layer**: Accepts preprocessed tabular features (43 features for IBM dataset).
2.  **Initial Processing**: Immediately after the input, the data must pass through a **Fully Connected Layer (128 units)** followed by a **Batch Normalization Layer**. This step normalizes the tabular distribution before the convolutional blocks [Figure 2].
3.  **Parallel Ensemble Bi-TCN Blocks**: The output from the initial BN layer branches into two parallel paths:
    *   **Path 1 (Bi-TCN Layer 1)**: 32 filters, Kernel size = 3, Dilation rates = ****.
    *   **Path 2 (Bi-TCN Layer 2)**: 64 filters, Kernel size = 5, Dilation rates = ****.
4.  **Concatenation**: The outputs from both parallel Bi-TCN paths are concatenated [Figure 2].
5.  **Output Processing**:
    *   Flatten Layer.
    *   Final Fully Connected layers with **Dropout (rate = 0.5)**.
    *   Softmax/Sigmoid output for binary classification (Attrition vs. No Attrition).

## 3. Specific Technical Requirements
*   **Dilation Rates**: Each Bi-TCN block must implement multiple dilation levels (1, 2, and 4) to expand the receptive field.
*   **Causality**: Maintain **Dilated Causal Convolutions** by ensuring padding is removed from the future side of the output tensor.
*   **Bidirectionality**: Process sequences in both forward and backward orientations (using `torch.flip` on the time dimension).
*   **Residual Connections**: Each block must include a residual link (with a 1x1 convolution if channel dimensions change) to stabilize training.

## 4. Current Implementation Issues (To be fixed)
*   **Wrong Sequence**: Current code puts FC/BN at the end; it should be at the beginning.
*   **Sequential vs Parallel**: Current code stacks layers (`bitcn1` -> `bitcn2`); it must be changed to parallel branches that are later concatenated.
*   **Missing Dilation**: Current code only uses dilation 1 and 2; it must support the full **** sequence.

## 5. Training Pipeline Context
*   **Dataset**: IBM HR Analytics (1470 records, 43 features after encoding).
*   **Augmentation**: Use **GAN-based data augmentation** inside the 5-fold Cross-Validation loop (Target: ~1500 samples per class).
*   **CV Strategy**: Must remain **Leakage-safe** (Augment and Scale only on the Training Fold, then test on original data).
*   **Hyperparameters**: Adam (lr=0.001), 50 Epochs, Batch size 128, L2 Weight Decay ($1e-5$).

---

**Next Step for AI Agent**: Please provide the refactored `BiTCN` class and the updated `forward` pass logic while maintaining the existing 5-fold CV training loop.

---
