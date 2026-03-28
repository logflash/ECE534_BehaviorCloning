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


# Maps (x_old_class, theta_old_class) → joint 5-class index.
# x_old_class: 0=-0.3, 1=0.0, 2=+0.3  (only 1 and 2 are valid here)
# theta_old_class: 0=-90, 1=0, 2=+90
_LABEL_MAP = {
    (1, 0): 0,  # x= 0.0, θ= -90
    (1, 2): 1,  # x= 0.0, θ= +90
    (2, 0): 2,  # x=+0.3, θ= -90
    (2, 1): 3,  # x=+0.3, θ=   0
    (2, 2): 4,  # x=+0.3, θ= +90
}
_CLASS_LABELS = ["x=0 θ=-90", "x=0 θ=+90", "x=.3 θ=-90", "x=.3 θ=0", "x=.3 θ=+90"]


def build_5class_labels(raw_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert raw 3-column labels to 5-class joint labels, filtering invalid rows.

    Valid rows: x_class ∈ {1, 2} (non-negative x) and not (x=0, theta=0).
    Returns (mask, y) where mask selects valid rows and y is shape (M,) int64.
    """
    x_cls = raw_labels[:, 0]
    theta_cls = raw_labels[:, 2]
    mask = np.array([(xc, tc) in _LABEL_MAP for xc, tc in zip(x_cls, theta_cls)])
    y = np.array([_LABEL_MAP[(xc, tc)]
                  for xc, tc in zip(x_cls[mask], theta_cls[mask])], dtype=np.int64)
    return mask, y


def plot_history(history: dict, output_path: str) -> None:
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["val_acc"], color="steelblue")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy (joint 5-class)")
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
    X_all = data["images"]   # (N, 84, 84, 3), uint8
    raw_labels = data["labels"]  # (N, 3), int64

    # ── Build 5-class labels, filtering invalid examples ─────────────────────
    mask, y = build_5class_labels(raw_labels)
    X = X_all[mask]
    n_dropped = len(X_all) - len(X)
    print(f"  {len(X)} valid examples ({n_dropped} dropped: negative x or x=0,θ=0)")
    print(f"  images: {X.shape}  {X.dtype}")
    print(f"  labels: {y.shape}  {y.dtype}  (5-class joint)")

    # ── Train / test split (stratified by joint class) ────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
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
        use_icw=args.icw,
        verbose=True,
    )

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    print("\n── Test set evaluation ──")
    metrics = policy.evaluate(X_test, y_test)
    print(f"  loss : {metrics['loss']:.4f}")
    print(f"  acc  : {metrics['acc']:.4f}")

    # ── Prediction distribution (verify model isn't collapsed) ────────────────
    preds = policy.predict(X_test)
    print("\n── Predicted class distribution on test set ──")
    counts = np.bincount(preds, minlength=5)
    total = counts.sum()
    for i, label in enumerate(_CLASS_LABELS):
        print(f"  class {i} ({label}): {counts[i]:4d} ({100*counts[i]/total:4.1f}%)")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "cnn_policy.pth")
    policy.save(model_path)

    # ── Plot training curves ──────────────────────────────────────────────────
    curve_path = os.path.join(args.output_dir, "training_curve.png")
    plot_history(history, curve_path)


if __name__ == "__main__":
    main()
