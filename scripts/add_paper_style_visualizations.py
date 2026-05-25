"""
Script to add paper-style Loss and Accuracy visualizations to the notebook.
This script generates code cells to insert into the notebook's visualization section.
"""

paper_style_viz_code = '''
# Calculate mean metrics across folds for paper-style presentation
all_epochs = []
all_train_loss = []
all_val_loss = []
all_train_accuracy = []
all_val_accuracy = []

for fold, hist in histories.items():
    all_epochs.append(hist["epoch"].values)
    all_train_loss.append(hist["train_loss"].values)
    all_val_loss.append(hist["val_loss"].values)
    all_train_accuracy.append(hist["train_accuracy"].values)
    all_val_accuracy.append(hist["val_accuracy"].values)

# Pad to same length and compute mean across folds
max_epochs = max(len(e) for e in all_epochs)
def pad_array(arr, max_len):
    padded = np.full(max_len, np.nan)
    padded[:len(arr)] = arr
    return padded

train_loss_padded = np.array([pad_array(loss, max_epochs) for loss in all_train_loss])
val_loss_padded = np.array([pad_array(loss, max_epochs) for loss in all_val_loss])
train_acc_padded = np.array([pad_array(acc, max_epochs) for acc in all_train_accuracy])
val_acc_padded = np.array([pad_array(acc, max_epochs) for acc in all_val_accuracy])

mean_train_loss = np.nanmean(train_loss_padded, axis=0)
mean_val_loss = np.nanmean(val_loss_padded, axis=0)
std_train_loss = np.nanstd(train_loss_padded, axis=0)
std_val_loss = np.nanstd(val_loss_padded, axis=0)

mean_train_acc = np.nanmean(train_acc_padded, axis=0)
mean_val_acc = np.nanmean(val_acc_padded, axis=0)
std_train_acc = np.nanstd(train_acc_padded, axis=0)
std_val_acc = np.nanstd(val_acc_padded, axis=0)

epochs_range = np.arange(1, max_epochs + 1)

# Paper-style Loss Diagram
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs_range, mean_train_loss, linewidth=2.5, color="#1f77b4", label="Training Loss", marker="o", markersize=3, markevery=max(1, max_epochs // 15))
ax.fill_between(epochs_range, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, 
                 alpha=0.2, color="#1f77b4")
ax.plot(epochs_range, mean_val_loss, linewidth=2.5, color="#ff7f0e", label="Validation Loss", marker="s", markersize=3, markevery=max(1, max_epochs // 15))
ax.fill_between(epochs_range, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, 
                 alpha=0.2, color="#ff7f0e")
ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
ax.set_title("Bi-TCN Training: Loss and Validation Loss", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "paper_loss_curves.png", dpi=300, bbox_inches="tight")
plt.show()

# Paper-style Accuracy Diagram
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs_range, mean_train_acc, linewidth=2.5, color="#2ca02c", label="Training Accuracy", marker="o", markersize=3, markevery=max(1, max_epochs // 15))
ax.fill_between(epochs_range, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, 
                 alpha=0.2, color="#2ca02c")
ax.plot(epochs_range, mean_val_acc, linewidth=2.5, color="#d62728", label="Validation Accuracy", marker="s", markersize=3, markevery=max(1, max_epochs // 15))
ax.fill_between(epochs_range, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, 
                 alpha=0.2, color="#d62728")
ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
ax.set_title("Bi-TCN Training: Accuracy and Validation Accuracy", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "paper_accuracy_curves.png", dpi=300, bbox_inches="tight")
plt.show()

print("Paper-style visualizations saved:")
print(f"  - {FIGURE_DIR / 'paper_loss_curves.png'}")
print(f"  - {FIGURE_DIR / 'paper_accuracy_curves.png'}")
'''

print("Script to add paper-style Loss and Accuracy visualizations")
print("=" * 60)
print("\nThe following code will be added to the notebook after the Visualization section:")
print("\n" + paper_style_viz_code)
print("\n" + "=" * 60)
print("\nThis script will:")
print("1. Calculate mean Loss/Accuracy across all 5 folds with std bands")
print("2. Create paper-style Loss diagram with training vs validation")
print("3. Create paper-style Accuracy diagram with training vs validation")
print("4. Save both figures as PNG with high resolution (300 dpi)")
print("\nThe visualizations match academic paper presentation style.")
