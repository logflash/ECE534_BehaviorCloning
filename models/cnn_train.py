"""
cnn_train.py

Train the CNN behavior cloning policy on the preprocessed dataset.

Usage:
    python models/cnn_train.py
    python models/cnn_train.py --data ./data/dataset.npz --epochs 30 --lr 1e-3
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Allow running from repo root or from models/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_policy import CNNPolicy


def plot_history(history: dict, output_path: str) -> None:
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy curves
    axes[1].plot(history["val_acc_x"],     label="x.vel")
    axes[1].plot(history["val_acc_theta"], label="θ.vel")
    axes[1].plot(history["val_acc_mean"],  label="mean", linestyle="--", color="black")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy per Action Component")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Training curve saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CNN behavior cloning policy")
    parser.add_argument("--data",       default="./data/dataset.npz",
                        help="Path to dataset .npz file (default: ./data/dataset.npz)")
    parser.add_argument("--epochs",     type=int,   default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int,   default=64,
                        help="Mini-batch size (default: 64)")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--val-split",  type=float, default=0.1,
                        help="Validation fraction within training set (default: 0.1)")
    parser.add_argument("--test-size",  type=float, default=0.15,
                        help="Held-out test fraction (default: 0.15)")
    parser.add_argument("--output-dir", default="./models/checkpoints",
                        help="Directory for saved model and plots (default: ./models/checkpoints)")
    parser.add_argument("--device",     default=None,
                        help="'cuda' or 'cpu' (default: auto-detect)")
    parser.add_argument("--nostop",     action="store_true",
                        help="Discard examples where x.vel==0 AND theta.vel==0 during training")
    parser.add_argument("--icw",        action="store_true",
                        help="Use inverted class weights to counteract class imbalance")
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
    if not os.path.exists(args.data):
        print(f"Dataset not found: {args.data}")
        print("Run  python dataset/construct_dataset.py  first.")
        sys.exit(1)

    print(f"Loading dataset from {args.data} ...")
    data = np.load(args.data)
    X = data["images"]              # (N, 84, 84, 3), uint8
    y = data["labels"][:, [0, 2]]  # (N, 2), int64 — x.vel and theta.vel only

    print(f"  images: {X.shape}  {X.dtype}")
    print(f"  labels: {y.shape}  {y.dtype}  (x.vel, theta.vel)")

    # ── Train / test split (stratified by joint x.vel + theta.vel label) ─────
    strat_key = y[:, 0] * 3 + y[:, 1]  # unique int per (x_class, theta_class)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=strat_key
    )
    print(f"\nSplit: {len(X_train)} train / {len(X_test)} test")

    # ── Instantiate and train ─────────────────────────────────────────────────
    policy = CNNPolicy(lr=args.lr, device=args.device)
    print(f"\nTraining on device: {policy.device}")

    history = policy.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        filter_stop=args.nostop,
        use_icw=args.icw,
        verbose=True,
    )

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    print("\n── Test set evaluation ──")
    metrics = policy.evaluate(X_test, y_test)
    print(f"  loss      : {metrics['loss']:.4f}")
    print(f"  acc x.vel : {metrics['acc_x']:.4f}")
    print(f"  acc θ.vel : {metrics['acc_theta']:.4f}")
    print(f"  acc mean  : {metrics['acc_mean']:.4f}")

    # ── Prediction distribution (verify model isn't collapsed to all-stop) ────
    preds = policy.predict(X_test)
    print("\n── Predicted class distribution on test set ──")
    for col, (name, values) in enumerate([("x.vel", [-0.3, 0.0, 0.3]), ("θ.vel", [-90, 0, 90])]):
        counts = np.bincount(preds[:, col], minlength=3)
        total = counts.sum()
        row = "  ".join(f"{v:>5}: {counts[i]:4d} ({100*counts[i]/total:4.1f}%)"
                        for i, v in enumerate(values))
        print(f"  {name}  {row}")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "cnn_policy.pth")
    policy.save(model_path)

    # ── Plot training curves ──────────────────────────────────────────────────
    curve_path = os.path.join(args.output_dir, "training_curve.png")
    plot_history(history, curve_path)


if __name__ == "__main__":
    main()
