"""
Binary Classification with PyTorch — A Complete Walkthrough
============================================================

This script demonstrates how to build, train, and evaluate a binary classifier
using PyTorch. We use the real-world Breast Cancer Wisconsin dataset:

  - Task   : Predict whether a tumour is Malignant (0) or Benign (1)
  - Source : sklearn.datasets.load_breast_cancer
  - Samples: 569 patients
  - Features: 30 numeric measurements (e.g. radius, texture, perimeter, area …)

Key concepts covered:
  1. Loading a real-world dataset and normalising features
  2. Splitting into train / test sets
  3. Converting NumPy arrays → PyTorch tensors → DataLoaders
  4. Defining a feedforward neural network (nn.Module)
  5. Choosing a loss function (BCEWithLogitsLoss) and optimizer (Adam)
  6. Training loop with epoch-level metrics
  7. Evaluation on the test set (accuracy, precision, recall, F1)
  8. Visualising loss, accuracy and the confusion matrix
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend so it works headless
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────────
# 1. LOAD & EXPLORE THE DATASET
# ──────────────────────────────────────────────
# The Breast Cancer Wisconsin dataset contains measurements taken from
# digitised images of fine needle aspirate (FNA) biopsies.
#
# Each row is one patient.  The 30 features describe cell-nucleus properties
# such as radius, texture, perimeter, area, smoothness, etc.
#
# Label:
#   0 → Malignant (cancerous)
#   1 → Benign    (non-cancerous)

data = load_breast_cancer()
X, y = data.data, data.target          # X: (569, 30),  y: (569,)
feature_names = data.feature_names     # e.g. "mean radius", "mean texture", …
target_names  = data.target_names      # ["malignant", "benign"]

print("=" * 55)
print("Dataset : Breast Cancer Wisconsin")
print(f"Samples : {X.shape[0]}")
print(f"Features: {X.shape[1]}  ({', '.join(feature_names[:3])}, …)")
print(f"Classes : 0 = {target_names[0]}, 1 = {target_names[1]}")
print(f"Class distribution — Malignant: {(y==0).sum()}  Benign: {(y==1).sum()}")
print("=" * 55, "\n")

# ──────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT  +  FEATURE SCALING
# ──────────────────────────────────────────────
# Real-world features live on wildly different scales:
#   "mean radius"  ≈ 6–28       "mean area" ≈ 140–2500
#
# StandardScaler transforms each feature to mean=0, std=1.
# IMPORTANT: fit the scaler ONLY on training data, then apply it to test data.
# Fitting on test data would leak information (data leakage).

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # stratify keeps class ratio equal
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # learn mean/std from train set
X_test  = scaler.transform(X_test)        # apply same transform to test set

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print(f"Features per sample: {X_train.shape[1]}\n")

# ──────────────────────────────────────────────
# 3. CONVERT TO PYTORCH TENSORS & DATALOADERS
# ──────────────────────────────────────────────
# PyTorch works with its own Tensor type. We also wrap our data in a
# DataLoader so we can iterate in mini-batches during training.

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # shape (N, 1)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ──────────────────────────────────────────────
# 4. DEFINE THE MODEL
# ──────────────────────────────────────────────
# Feedforward network with two hidden layers, ReLU activations, and a
# single output neuron (raw logit — sigmoid is inside the loss function).
#
# Architecture:
#   Input (30) → Hidden (64) → ReLU → Hidden (32) → ReLU → Output (1)

INPUT_DIM = X_train.shape[1]   # 30 features


class BinaryClassifier(nn.Module):
    """Feedforward network for binary classification."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),   # first hidden layer
            nn.ReLU(),
            nn.Linear(64, 32),          # second hidden layer
            nn.ReLU(),
            nn.Linear(32, 1),           # output layer — raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns raw logits (not probabilities)."""
        return self.network(x)


model = BinaryClassifier(input_dim=INPUT_DIM)
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ──────────────────────────────────────────────
# 5. LOSS FUNCTION & OPTIMISER
# ──────────────────────────────────────────────
# BCEWithLogitsLoss combines a Sigmoid layer and Binary Cross-Entropy loss
# in one numerically stable operation.  This is preferred over applying
# sigmoid manually then using BCELoss.
#
# Adam is a widely-used adaptive learning-rate optimiser.

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ──────────────────────────────────────────────
# 6. TRAINING LOOP
# ──────────────────────────────────────────────
# Each epoch:
#   a) Iterate over mini-batches
#   b) Forward pass  → compute predictions
#   c) Compute loss  → BCEWithLogitsLoss(predictions, labels)
#   d) Backward pass → compute gradients
#   e) Optimiser step → update weights
#   f) Zero gradients for the next batch

NUM_EPOCHS = 100
history = {"loss": [], "accuracy": []}

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss  += loss.item() * X_batch.size(0)
        predictions  = (torch.sigmoid(logits) >= 0.5).float()
        correct     += (predictions == y_batch).sum().item()
        total       += y_batch.size(0)

    avg_loss = epoch_loss / total
    accuracy = correct / total
    history["loss"].append(avg_loss)
    history["accuracy"].append(accuracy)

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  |  Loss: {avg_loss:.4f}  |  Accuracy: {accuracy:.4f}")

print()

# ──────────────────────────────────────────────
# 7. EVALUATE ON THE TEST SET
# ──────────────────────────────────────────────

model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        logits = model(X_batch)
        preds  = (torch.sigmoid(logits) >= 0.5).float()
        all_preds.append(preds)
        all_labels.append(y_batch)

all_preds  = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

test_accuracy = (all_preds == all_labels).mean()
print(f"Test Accuracy: {test_accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Malignant", "Benign"]))

# ──────────────────────────────────────────────
# 8. VISUALISATIONS
# ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Breast Cancer Classifier — Training Results", fontsize=15, fontweight="bold")

# --- 8a. Training loss curve ---
axes[0].plot(history["loss"], color="steelblue", linewidth=2)
axes[0].set_title("Training Loss over Epochs", fontsize=13)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCE Loss")
axes[0].grid(True, alpha=0.3)

# --- 8b. Training accuracy curve ---
axes[1].plot(history["accuracy"], color="darkorange", linewidth=2)
axes[1].set_title("Training Accuracy over Epochs", fontsize=13)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0.5, 1.0)
axes[1].grid(True, alpha=0.3)

# --- 8c. Confusion matrix ---
# Rows = actual class, Columns = predicted class.
# Diagonal = correctly classified.  Off-diagonal = mistakes.
#
#                  Predicted
#              Malignant  Benign
# Actual  Malignant  TN     FP   ← False Positive: told "Benign" but actually Malignant (dangerous!)
#         Benign     FN     TP   ← False Negative: told "Malignant" but actually Benign

cm = confusion_matrix(all_labels, all_preds)
im = axes[2].imshow(cm, cmap="Blues")
axes[2].set_xticks([0, 1])
axes[2].set_yticks([0, 1])
axes[2].set_xticklabels(["Malignant", "Benign"], fontsize=11)
axes[2].set_yticklabels(["Malignant", "Benign"], fontsize=11)
axes[2].set_xlabel("Predicted Label", fontsize=12)
axes[2].set_ylabel("True Label", fontsize=12)
axes[2].set_title("Confusion Matrix (test set)", fontsize=13)
fig.colorbar(im, ax=axes[2])

for i in range(2):
    for j in range(2):
        axes[2].text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
print("Saved training_results.png")

# ──────────────────────────────────────────────
# 9. SAVING & LOADING THE MODEL
# ──────────────────────────────────────────────
# Best practice: save the state_dict (just the learned parameters),
# not the entire model object.

torch.save(model.state_dict(), "binary_classifier.pth")
print("Saved model weights to binary_classifier.pth")

# To reload later:
#   loaded_model = BinaryClassifier(input_dim=30)
#   loaded_model.load_state_dict(torch.load("binary_classifier.pth"))
#   loaded_model.eval()
