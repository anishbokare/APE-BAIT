#!/usr/bin/env python
"""
APE-BAIT Surrogate Model Training Script
Trains IDS, malware, and anomaly surrogate models on synthetic data.
Exports trained weights to data/models/ directory.

Usage:
    python scripts/train_surrogate.py --epochs 50 --output data/models/
    python scripts/train_surrogate.py --model ids --epochs 100
    python scripts/train_surrogate.py --use-pretrained
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.core.logger import get_logger
from src.models.model_zoo import ModelZoo
from src.core.config import load_config


logger = get_logger("ape-bait.train")


# ─── Synthetic Data Generation ───────────────────────────────────

def generate_ids_data(n_samples: int = 10000, feature_dim: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic IDS training data.
    
    Benign traffic: lower entropy, consistent port patterns
    Malicious traffic: higher variance, unusual feature patterns
    """
    np.random.seed(42)
    n_benign = n_samples // 2
    n_malicious = n_samples - n_benign

    # Benign: low-variance, structured features
    benign = np.random.beta(2, 5, (n_benign, feature_dim)).astype(np.float32)
    # Payload bytes (positions 8:136) tend to be text-like (lower values)
    benign[:, 8:136] = np.random.beta(2, 8, (n_benign, 128)).astype(np.float32)

    # Malicious: higher variance, unusual patterns
    malicious = np.random.beta(0.5, 0.5, (n_malicious, feature_dim)).astype(np.float32)
    malicious[:, 8:136] = np.random.uniform(0, 1, (n_malicious, 128)).astype(np.float32)
    # Simulate exploit patterns: high entropy payload
    malicious[:, 8:50] = np.random.uniform(0.7, 1.0, (n_malicious, 42)).astype(np.float32)

    X = np.vstack([benign, malicious])
    y = np.array([0] * n_benign + [1] * n_malicious, dtype=np.int64)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def generate_malware_data(n_samples: int = 8000, feature_dim: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic malware classification data (9 classes: 1 benign + 8 families)."""
    np.random.seed(123)
    n_classes = 9
    n_per_class = n_samples // n_classes
    
    all_X, all_y = [], []
    for cls in range(n_classes):
        # Each class has slightly different feature distribution
        mean = np.random.uniform(0.2, 0.8, feature_dim)
        std = np.random.uniform(0.05, 0.2, feature_dim)
        samples = np.random.normal(mean, std, (n_per_class, feature_dim))
        samples = np.clip(samples, 0, 1).astype(np.float32)
        all_X.append(samples)
        all_y.extend([cls] * n_per_class)
    
    X = np.vstack(all_X)
    y = np.array(all_y, dtype=np.int64)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def generate_anomaly_data(n_samples: int = 8000, feature_dim: int = 256) -> np.ndarray:
    """Generate 'normal' traffic data for autoencoder training."""
    np.random.seed(999)
    # Normal traffic: structured, low-variance features
    X = np.random.beta(2, 5, (n_samples, feature_dim)).astype(np.float32)
    X[:, 8:136] = np.random.beta(3, 7, (n_samples, 128)).astype(np.float32)
    return X


# ─── Training Functions ───────────────────────────────────────────

def train_classifier(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    val_split: float = 0.1,
) -> dict:
    """Generic classifier training loop."""
    model = model.to(device)
    
    # Train/val split
    n_val = int(len(X) * val_split)
    X_train, y_train = X[n_val:], y[n_val:]
    X_val, y_val = X[:n_val], y[:n_val]
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        
        train_loss /= len(X_train)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item() * len(X_batch)
                correct += (output.argmax(1) == y_batch).sum().item()
        
        val_loss /= len(X_val)
        val_acc = correct / len(X_val)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if epoch % 10 == 0 or epoch == epochs:
            logger.info(
                f"Epoch {epoch}/{epochs}",
                train_loss=round(train_loss, 4),
                val_loss=round(val_loss, 4),
                val_acc=round(val_acc, 4),
            )
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    
    history["best_val_acc"] = best_val_acc
    return history


def train_autoencoder(
    model: nn.Module,
    X: np.ndarray,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict:
    """Autoencoder training for anomaly detector."""
    model = model.to(device)
    
    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {"train_loss": []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(X_batch)
            loss = criterion(x_hat, X_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        
        epoch_loss /= len(X)
        history["train_loss"].append(epoch_loss)
        
        if epoch % 10 == 0 or epoch == epochs:
            logger.info(f"Autoencoder Epoch {epoch}/{epochs}", loss=round(epoch_loss, 6))
    
    return history


# ─── Main Script ──────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train APE-BAIT surrogate models")
    parser.add_argument("--model", choices=["ids", "malware", "anomaly", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="data/models/")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--use-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(config_path=args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device
    feature_dim = cfg.models.feature_dim
    
    if args.use_pretrained:
        logger.info("Using pre-trained weights (if available)")
        zoo = ModelZoo(cfg)
        zoo.preload_all()
        logger.info("Models loaded successfully")
        return
    
    start_time = time.time()
    
    # ── Train IDS surrogate ─────────────────────────────────────
    if args.model in ("ids", "all"):
        logger.info("Training SurrogateIDS...")
        from src.models.surrogate_ids import build_surrogate_ids
        
        X, y = generate_ids_data(args.samples, feature_dim)
        model = build_surrogate_ids(input_dim=feature_dim, device=device)
        
        history = train_classifier(
            model, X, y,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        
        out_path = output_dir / "surrogate_ids.pt"
        torch.save(model.state_dict(), out_path)
        logger.info(
            f"IDS model saved to {out_path}",
            best_val_acc=round(history["best_val_acc"], 4),
        )
    
    # ── Train Malware surrogate ─────────────────────────────────
    if args.model in ("malware", "all"):
        logger.info("Training SurrogateMalwareClassifier...")
        from src.models.surrogate_malware import build_surrogate_malware
        
        X, y = generate_malware_data(args.samples, feature_dim)
        model = build_surrogate_malware(input_dim=feature_dim, device=device)
        
        history = train_classifier(
            model, X, y,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        
        out_path = output_dir / "surrogate_malware.pt"
        torch.save(model.state_dict(), out_path)
        logger.info(
            f"Malware model saved to {out_path}",
            best_val_acc=round(history["best_val_acc"], 4),
        )
    
    # ── Train Anomaly detector ──────────────────────────────────
    if args.model in ("anomaly", "all"):
        logger.info("Training SurrogateAnomalyDetector...")
        from src.models.surrogate_anomaly import build_surrogate_anomaly
        
        X = generate_anomaly_data(args.samples, feature_dim)
        model = build_surrogate_anomaly(input_dim=feature_dim, device=device)
        
        history = train_autoencoder(
            model, X,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        
        out_path = output_dir / "surrogate_anomaly.pt"
        torch.save(model.state_dict(), out_path)
        logger.info(f"Anomaly model saved to {out_path}")
    
    elapsed = time.time() - start_time
    logger.info(
        f"Training complete in {elapsed:.1f}s",
        output_dir=str(output_dir),
    )
    print(f"\n✅ Training complete! Models saved to {output_dir}")


if __name__ == "__main__":
    main()
