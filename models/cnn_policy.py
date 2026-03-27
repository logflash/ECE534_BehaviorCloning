"""
cnn_policy.py

CNN-based behavior cloning policy for predicting discretized velocity commands
from robot camera images.

The model outputs 9 logits representing 3 independent softmax distributions:
  - logits[0:3]: x.vel  ∈ {-0.3, 0.0, +0.3}  → class indices {0, 1, 2}
  - logits[3:6]: y.vel  ∈ {-0.3, 0.0, +0.3}  → class indices {0, 1, 2}
  - logits[6:9]: θ.vel  ∈ { -90,   0,  +90}  → class indices {0, 1, 2}
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ──────────────────────────────────────────────────────────────────────────────
# Neural network module
# ──────────────────────────────────────────────────────────────────────────────

class _ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        super().__init__(*layers)


class _CNNNet(nn.Module):
    """
    Small CNN for 84×84 RGB input → 9 logits.

    Architecture:
        Block 1: Conv(3→32)  + BN + ReLU + MaxPool  → (B, 32, 42, 42)
        Block 2: Conv(32→64) + BN + ReLU + MaxPool  → (B, 64, 21, 21)
        Block 3: Conv(64→128)+ BN + ReLU + MaxPool  → (B,128, 10, 10)
        Block 4: Conv(128→256)+ BN + ReLU           → (B,256, 10, 10)
        GlobalAvgPool                               → (B, 256)
        Dropout(0.3)
        Linear(256 → 9)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(3,   32,  pool=True),
            _ConvBlock(32,  64,  pool=True),
            _ConvBlock(64,  128, pool=True),
            _ConvBlock(128, 256, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# ──────────────────────────────────────────────────────────────────────────────
# Policy wrapper
# ──────────────────────────────────────────────────────────────────────────────

class CNNPolicy:
    """
    Behavior cloning policy backed by a small CNN.

    Parameters
    ----------
    lr : float
        Initial learning rate for Adam optimizer.
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(self, lr: float = 1e-3, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.lr = lr

        self.model = _CNNNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self._ce = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, X: np.ndarray) -> torch.Tensor:
        """Convert (N, H, W, 3) uint8 numpy → (N, 3, H, W) float32 tensor in [0,1]."""
        t = torch.from_numpy(X).float() / 255.0   # (N, H, W, 3)
        return t.permute(0, 3, 1, 2)              # (N, 3, H, W)

    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sum of cross-entropy losses for the 3 action components."""
        return (
            self._ce(logits[:, 0:3], y[:, 0])
            + self._ce(logits[:, 3:6], y[:, 1])
            + self._ce(logits[:, 6:9], y[:, 2])
        )

    def _accuracy(self, logits: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
        """Per-component accuracy: (acc_x, acc_y, acc_theta)."""
        pred_x = logits[:, 0:3].argmax(dim=1)
        pred_y = logits[:, 3:6].argmax(dim=1)
        pred_t = logits[:, 6:9].argmax(dim=1)
        acc_x = (pred_x == y[:, 0]).float().mean().item()
        acc_y = (pred_y == y[:, 1]).float().mean().item()
        acc_t = (pred_t == y[:, 2]).float().mean().item()
        return acc_x, acc_y, acc_t

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        val_split: float = 0.1,
        verbose: bool = True,
    ) -> dict:
        """
        Train the policy on image-action pairs.

        Parameters
        ----------
        X : np.ndarray, shape (N, H, W, 3), dtype uint8
            Image observations.
        y : np.ndarray, shape (N, 3), dtype int64
            Class labels for [x.vel, y.vel, theta.vel].
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size.
        val_split : float
            Fraction of data held out for validation (shuffled).
        verbose : bool
            Print epoch summaries.

        Returns
        -------
        dict with keys 'train_loss', 'val_loss', 'val_acc_x', 'val_acc_y',
        'val_acc_theta', 'val_acc_mean' — each a list of per-epoch values.
        """
        N = len(X)
        n_val = max(1, int(N * val_split))
        perm = np.random.permutation(N)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]

        X_train_t = self._preprocess(X_train)
        y_train_t = torch.from_numpy(y_train).long()
        X_val_t   = self._preprocess(X_val)
        y_val_t   = torch.from_numpy(y_val).long()

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=batch_size, shuffle=True, pin_memory=(self.device.type == "cuda"),
        )

        history: dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "val_acc_x": [], "val_acc_y": [], "val_acc_theta": [], "val_acc_mean": [],
        }

        best_val_loss = float("inf")
        best_weights = None

        for epoch in range(1, epochs + 1):
            # ── training ──
            self.model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self._loss(logits, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(xb)

            train_loss = total_loss / len(X_train)

            # ── validation ──
            self.model.eval()
            with torch.no_grad():
                val_logits = self._infer_batched(X_val_t, batch_size=256)
                val_loss = self._loss(val_logits, y_val_t.to(self.device)).item()
                acc_x, acc_y, acc_t = self._accuracy(val_logits, y_val_t.to(self.device))
                acc_mean = (acc_x + acc_y + acc_t) / 3

            # ── checkpoint best weights ──
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                marker = " *"
            else:
                marker = ""

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc_x"].append(acc_x)
            history["val_acc_y"].append(acc_y)
            history["val_acc_theta"].append(acc_t)
            history["val_acc_mean"].append(acc_mean)

            if verbose:
                print(
                    f"Epoch {epoch:3d}/{epochs}  "
                    f"train_loss={train_loss:.4f}  "
                    f"val_loss={val_loss:.4f}  "
                    f"val_acc=[x:{acc_x:.3f} y:{acc_y:.3f} θ:{acc_t:.3f}]  "
                    f"mean={acc_mean:.3f}{marker}"
                )

        # ── restore best weights ──
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            if verbose:
                best_epoch = int(np.argmin(history["val_loss"])) + 1
                print(f"Restored best weights from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

        return history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Compute loss and per-component accuracy on a held-out set.

        Parameters
        ----------
        X : np.ndarray, shape (N, H, W, 3), dtype uint8
        y : np.ndarray, shape (N, 3), dtype int64

        Returns
        -------
        dict with keys: 'loss', 'acc_x', 'acc_y', 'acc_theta', 'acc_mean'
        """
        X_t = self._preprocess(X)
        y_t = torch.from_numpy(y).long()

        self.model.eval()
        with torch.no_grad():
            logits = self._infer_batched(X_t, batch_size=256)
            loss = self._loss(logits, y_t.to(self.device)).item()
            acc_x, acc_y, acc_t = self._accuracy(logits, y_t.to(self.device))

        return {
            "loss": loss,
            "acc_x": acc_x,
            "acc_y": acc_y,
            "acc_theta": acc_t,
            "acc_mean": (acc_x + acc_y + acc_t) / 3,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class indices for each action component.

        Returns
        -------
        np.ndarray of shape (N, 3), dtype int64
        """
        X_t = self._preprocess(X)
        self.model.eval()
        with torch.no_grad():
            logits = self._infer_batched(X_t, batch_size=256)  # (N, 9)
        preds = torch.stack([
            logits[:, 0:3].argmax(dim=1),
            logits[:, 3:6].argmax(dim=1),
            logits[:, 6:9].argmax(dim=1),
        ], dim=1)
        return preds.cpu().numpy().astype(np.int64)

    # Maps class index → real control value for each component
    _XY_VALUES    = {0: -0.3, 1: 0.0, 2: 0.3}
    _THETA_VALUES = {0: -90.0, 1: 0.0, 2: 90.0}

    def get_controls(self, image: np.ndarray) -> dict[str, float]:
        """
        Run inference on a single image and return real-valued controls.

        Parameters
        ----------
        image : np.ndarray, shape (H, W, 3), dtype uint8

        Returns
        -------
        dict with keys 'x_vel', 'y_vel', 'theta_vel' mapped to their
        real control values: x/y ∈ {-0.3, 0.0, 0.3}, theta ∈ {-90, 0, 90}.
        """
        X = image[np.newaxis]  # (1, H, W, 3)
        classes = self.predict(X)[0]  # (3,)
        return {
            "x_vel":     self._XY_VALUES[int(classes[0])],
            "y_vel":     self._XY_VALUES[int(classes[1])],
            "theta_vel": self._THETA_VALUES[int(classes[2])],
        }

    def _infer_batched(self, X_t: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        """Run inference in batches to avoid OOM on large inputs."""
        results = []
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i : i + batch_size].to(self.device)
            results.append(self.model(xb))
        return torch.cat(results, dim=0)

    def save(self, path: str) -> None:
        """Save model weights and optimizer state to a .pth file."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "lr": self.lr,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights and optimizer state from a .pth file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"Model loaded from {path}")
