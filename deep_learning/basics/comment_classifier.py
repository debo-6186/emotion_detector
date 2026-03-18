import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torch.optim import AdamW

from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup
)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import pandas as pd                   # Data manipulation
import numpy as np                    # Numerical operations
from tqdm import tqdm                 # Progress bars
import os
import random
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)                          # Python's built-in random
    np.random.seed(seed)                       # NumPy's random
    torch.manual_seed(seed)                    # PyTorch CPU random
    torch.cuda.manual_seed_all(seed)           # PyTorch GPU random (all GPUs)
    torch.backends.cudnn.deterministic = True  # Make CUDA operations deterministic
    torch.backends.cudnn.benchmark = False     # Disable auto-tuning (for reproducibility)

set_seed(42)

class Config:
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    NUM_FOLDS = 5
    FOCAL_LOSS_GAMMA = 2.0

    LABEL_MAP = {
        "Insult": 0,
        "Threat": 1,
        "Sexual Harassment": 2,
    }

    NUM_CLASSES = len(LABEL_MAP)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CommentDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: AutoTokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

"""
THE PROBLEM WITH STANDARD CROSS-ENTROPY:
-----------------------------------------
Imagine your dataset is: 70% Insult, 20% Threat, 10% Sexual Harassment.
 
Standard cross-entropy treats every example equally. So the model learns to
be really good at "Insult" (because there are so many examples) but struggles
with "Sexual Harassment" (too few examples to learn the patterns).
 
Class weighting partially fixes this by multiplying the loss of rare classes
by a larger number. But it still treats easy and hard examples the same.
 
FOCAL LOSS — THE BETTER SOLUTION:
----------------------------------
Focal Loss (Lin et al., 2017 — originally for object detection) adds a
"focusing factor" that automatically reduces the loss for well-classified
(easy) examples and increases it for misclassified (hard) examples.
 
The formula is:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
 
Where:
    p_t    = model's predicted probability for the TRUE class
    gamma  = focusing parameter (we use 2.0)
    alpha  = class weight (inversely proportional to frequency)
 
INTUITION:
    - If the model is CONFIDENT and CORRECT (p_t ≈ 1.0):
      (1 - 1.0)^2 = 0  →  loss is essentially zero (ignore easy examples)
 
    - If the model is WRONG (p_t ≈ 0.0):
      (1 - 0.0)^2 = 1  →  full loss (focus on hard examples)
 
    - If the model is UNCERTAIN (p_t ≈ 0.5):
      (1 - 0.5)^2 = 0.25  →  reduced but still significant loss
 
This means the model spends most of its learning capacity on the ambiguous
boundary cases — exactly where improvement matters most.
"""

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
 
        Parameters
        ----------
        logits : torch.Tensor, shape (batch_size, num_classes)
            Raw model outputs (before softmax)
        targets : torch.Tensor, shape (batch_size,)
            Ground-truth class indices
 
        Returns
        -------
        loss : torch.Tensor
            Scalar loss value
        """
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()

class Classifier(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        return self.classifier(cls_output)
    
"""
WHY CLASS WEIGHTS?
------------------
If your dataset has 7000 Insults, 2000 Threats, and 1000 Sexual Harassment
examples, the model will naturally bias toward predicting "Insult" because
it's the safest bet (right 70% of the time!).
 
Class weights counteract this by making errors on rare classes more expensive.
We use "inverse frequency" weighting:
 
    weight_i = total_samples / (num_classes * count_of_class_i)
 
For our example:
    weight_Insult  = 10000 / (3 * 7000) = 0.48  (low weight — plenty of data)
    weight_Threat  = 10000 / (3 * 2000) = 1.67  (medium weight)
    weight_SexHar  = 10000 / (3 * 1000) = 3.33  (high weight — rare, so errors cost more)
 
These weights get passed into our Focal Loss as the alpha parameter.
"""

def compute_class_weights(labels: list) -> torch.Tensor:
    """
    Compute inverse-frequency class weights.
 
    Parameters
    ----------
    labels : list of int
        All training labels
 
    Returns
    -------
    weights : torch.Tensor, shape (num_classes,)
        Weight for each class
    """
    labels_array = np.array(labels)
    unique_classes = np.unique(labels_array)
    total = len(labels_array)
    n_classes = len(unique_classes)
 
    weights = []
    for cls in unique_classes:
        count = np.sum(labels_array == cls)
        weight = total / (n_classes * count)
        weights.append(weight)
        # Print for transparency
        print(f"  Class {cls}: {count} samples, weight = {weight:.3f}")
 
    return torch.tensor(weights, dtype=torch.float32)

"""
WHAT HAPPENS IN ONE TRAINING EPOCH:
-------------------------------------
For each batch of 16 comments:
 
1. FORWARD PASS: Feed comments through the model → get logits
2. COMPUTE LOSS: Compare logits to true labels using Focal Loss
3. BACKWARD PASS: Compute gradients (how much each weight contributed to error)
4. UPDATE WEIGHTS: Optimizer adjusts weights in the direction that reduces loss
5. UPDATE LEARNING RATE: Scheduler adjusts LR according to warmup schedule
6. REPEAT for all batches
 
The gradient flow looks like this:
 
    Loss ←── Focal Loss(logits, labels)
      ↓ backward()
    ∂Loss/∂W_classifier  →  update classifier weights
    ∂Loss/∂W_layer12     →  update DeBERTa layer 12
    ∂Loss/∂W_layer11     →  update DeBERTa layer 11
    ...                      (gradients flow through ALL layers)
    ∂Loss/∂W_layer1      →  update DeBERTa layer 1
 
Each weight gets updated as:  W_new = W_old - learning_rate * gradient
(Plus AdamW's momentum and adaptive learning rate adjustments)
"""

def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    dataloader: DataLoader,
    device: torch.device,
    num_epochs: int,
    verbose: bool = True
) -> float:
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    # tqdm wraps the dataloader to show a progress bar
    progress_bar = tqdm(dataloader, desc="  Training", leave=False)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / num_batches

"""
EVALUATION vs TRAINING:
-----------------------
During evaluation, we:
  - Set model.eval() → disables dropout
  - Use torch.no_grad() → disables gradient computation (saves memory + speed)
  - Do NOT update any weights
 
We collect all predictions and compare them to true labels to compute metrics.
"""

def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_epochs: int,
    verbose: bool = True
) -> float:
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            epoch_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            predictions = torch.argmax(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Collect predictions for metric computation
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(labels, predicted, average="macro")
    accuracy = correct / total
    return epoch_loss / len(dataloader), macro_f1, macro_f1, accuracy

"""
WHY K-FOLD CROSS-VALIDATION?
-----------------------------
With a single train/test split, your results depend heavily on WHICH examples
ended up in test. You might get lucky (easy test set) or unlucky (hard test set).
 
K-Fold solves this by rotating which portion is used for testing:
 
    Fold 1: [TEST] [Train] [Train] [Train] [Train]
    Fold 2: [Train] [TEST] [Train] [Train] [Train]
    Fold 3: [Train] [Train] [TEST] [Train] [Train]
    Fold 4: [Train] [Train] [Train] [TEST] [Train]
    Fold 5: [Train] [Train] [Train] [Train] [TEST]
 
Every example gets to be in the test set exactly once. The average performance
across all 5 folds is a much more reliable estimate of how the model will
perform on truly unseen data.
 
"Stratified" means each fold preserves the original class distribution.
If the full dataset is 70/20/10, each fold's train and test split will also
be approximately 70/20/10.
"""

def run_pipeline(df: pd.DataFrame, text_column: str, label_column: str) -> float:
    """
    Run the complete training pipeline with Stratified K-Fold CV.
 
    Parameters
    ----------
    df : pd.DataFrame
        Your dataset with at least two columns:
        - text_column: the raw comment strings
        - label_column: the category labels (e.g., "Insult", "Threat", "Sexual Harassment")
 
    text_column : str
        Name of the column containing the comment text
 
    label_column : str
        Name of the column containing the category labels
    """

    # Convert string labels to integers using our mapping
    df["label_encoded"] = df[label_column].map(Config.LABEL_MAP)

    # Check for unmapped labels (would be NaN)
    if df["label_encoded"].isna().any():
        unmapped = df[df["label_encoded"].isna()][label_column].unique()
        raise ValueError(f"Unknown labels found: {unmapped}. Expected: {list(Config.LABEL_MAP.keys())}")

    texts = df[text_column].values
    labels = df["label_encoded"].values.astype(int)

    print(f"  Total samples: {len(texts)}")
    print(f"  Label distribution:")
    for label_name, label_id in Config.LABEL_MAP.items():
        count = np.sum(labels == label_id)
        pct = 100 * count / len(labels)
        print(f"    {label_name} (class {label_id}): {count} ({pct:.1f}%)")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    class_weights = compute_class_weights(labels.tolist())

    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS,
        shuffle=True,     # Shuffle data before splitting
        random_state=42,  # For reproducibility
    )
    fold_results = []  # Store results from each fold
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
        print(f"  Fold {fold_idx + 1} / {Config.NUM_FOLDS}")
        print("  --------------------------------")

        # Split data into train and test sets
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        train_dataset = CommentDataset(train_texts, train_labels, tokenizer, Config.MAX_LENGTH)
        test_dataset = CommentDataset(test_texts, test_labels, tokenizer, Config.MAX_LENGTH)

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

        model = Classifier(Config.MODEL_NAME, Config.NUM_CLASSES).to(Config.DEVICE)
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_params = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": Config.WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,  # No decay for these parameters
            },
        ]
        optimizer = AdamW(optimizer_grouped_params, lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=Config.WARMUP_RATIO * len(train_loader), num_training_steps=Config.EPOCHS * len(train_loader))
        criterion = FocalLoss(alpha=class_weights, gamma=Config.FOCAL_LOSS_GAMMA)

        # ------ Initialize learning rate scheduler ------
        total_steps = len(train_loader) * Config.EPOCHS
        warmup_steps = int(total_steps * Config.WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        # ------ Train and evaluate the model ------
        best_macro_f1 = 0.0
        best_model = None
        for epoch in range(Config.EPOCHS):
            print(f"  Epoch {epoch + 1} / {Config.EPOCHS}")
            print("  ------------------------------")

            train_loss = train_one_epoch(model, criterion, optimizer, scheduler, train_loader, Config.DEVICE, Config.EPOCHS, verbose=True)
            test_loss, macro_f1, _, _ = evaluate(model, criterion, test_loader, Config.DEVICE, Config.EPOCHS, verbose=True)

            if test_loss["f1_macro"] > best_macro_f1:
                best_macro_f1 = test_loss["f1_macro"]
                best_model = model.state_dict()
                torch.save(best_model, f"best_model_fold{fold_idx + 1}.pth")
                print(f"  Best model saved for fold {fold_idx + 1}")

            fold_results.append({
                "fold": fold_idx + 1,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "macro_f1": macro_f1,
            })

        # ------ Save results ------
        results_df = pd.DataFrame(fold_results)
        results_df.to_csv(f"comment_classifier_results.csv", index=False)

"""
INFERENCE PIPELINE:
-------------------
Once training is complete, you'll want to classify new, unseen comments.
The inference function:
  1. Tokenizes the raw text (same as training)
  2. Passes it through the trained model
  3. Applies softmax to get probabilities
  4. Returns the predicted class AND confidence scores
 
This lets you set confidence thresholds — for example, only auto-moderate
comments where the model is > 90% confident, and flag uncertain ones for
human review.
"""

def predict(model: nn.Module, text: str, tokenizer: AutoTokenizer) -> dict:
    model.eval()
    label_names = list(Config.LABEL_MAP.keys())
    with torch.no_grad():
        encoding = tokenizer(text, add_special_tokens=True, max_length=Config.MAX_LENGTH, padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt")
        input_ids = encoding["input_ids"].to(Config.DEVICE)
        attention_mask = encoding["attention_mask"].to(Config.DEVICE)
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        return predicted_class, confidence
        