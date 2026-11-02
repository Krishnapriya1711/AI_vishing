"""Training and evaluation utilities for the LightweightCNN."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..evaluation.metrics import evaluate_predictions


# ============================================================
# TRAINING
# ============================================================

def train_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 10,
    device: str | None = None,
    verbose: bool = True,
) -> dict:
    """Train the CNN with cross-entropy loss and Adam optimiser.

    Returns
    -------
    dict with keys: train_losses, val_losses, best_val_loss, best_epoch
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    device_t = torch.device(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for specs, labels in train_loader:
            specs = specs.to(device_t)
            labels = labels.to(device_t)

            optimiser.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # --- Validation ---
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n_val = 0

            with torch.no_grad():
                for specs, labels in val_loader:
                    specs = specs.to(device_t)
                    labels = labels.to(device_t)
                    logits = model(specs)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    n_val += 1

            avg_val = val_loss / max(n_val, 1)
            val_losses.append(avg_val)

            if verbose:
                print(f"Epoch {epoch+1:3d}/{num_epochs}  "
                      f"train={avg_train:.4f}  val={avg_val:.4f}")

            # Early stopping
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_epoch = epoch + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1} "
                              f"(best={best_epoch})")
                    break
        else:
            if verbose:
                print(f"Epoch {epoch+1:3d}/{num_epochs}  train={avg_train:.4f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_cnn(
    model: nn.Module,
    test_loader: DataLoader,
    device: str | None = None,
) -> dict:
    """Evaluate the CNN on a test set.

    Returns
    -------
    dict from ``evaluate_predictions``
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    device_t = torch.device(device)
    model.eval()

    all_scores = []
    all_labels = []

    for specs, labels in test_loader:
        specs = specs.to(device_t)
        logits = model(specs)
        proba = torch.softmax(logits, dim=1)
        spoof_scores = proba[:, 1].cpu().numpy()
        all_scores.append(spoof_scores)
        all_labels.append(labels.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    return evaluate_predictions(labels, scores)


# ============================================================
# MODEL FACTORY
# ============================================================

def build_cnn(n_mels: int = 64, num_classes: int = 2) -> nn.Module:
    """Create a fresh LightweightCNN."""
    from .cnn import LightweightCNN
    return LightweightCNN(n_mels=n_mels, num_classes=num_classes)
