# Behavior Cloning w/ LeKiwi

CNN-based behavior cloning on the [`TagAggDann/lekiwi_green_block`](https://huggingface.co/datasets/TagAggDann/lekiwi_green_block) dataset (collected by us).
The model predicts discretized velocity commands (x, y, θ) from front-camera images.

---

## Setup

Create a Python 3.10 virtual environment named `rob534` and install all dependencies (including PyTorch with CUDA 12.4):

**PowerShell (Windows)**
```powershell
.\setup_venv.ps1
```

**Bash**
```bash
bash setup_venv.sh
```

**Zsh**
```zsh
zsh setup_venv.zsh
```

Activate the environment before running any scripts:

```powershell
# PowerShell
rob534\Scripts\Activate.ps1
```
```bash
# Bash / Zsh
source rob534/bin/activate
```

> If your GPU uses a different CUDA version, edit the `--extra-index-url` in the setup script:
> `cu124` → `cu121` (CUDA 12.1) or `cu118` (CUDA 11.8).

---

## 1. Build the dataset

Downloads the raw dataset from HuggingFace, decodes video frames, discretizes actions, and saves a compressed `.npz` file.

```bash
python dataset/construct_dataset.py
```

Output: `data/dataset.npz` (~50–150 MB compressed).

**Options**

| Flag | Default | Description |
|------|---------|-------------|
| `--output PATH` | `./data/dataset.npz` | Where to save the `.npz` file |
| `--raw-dir PATH` | `./data/raw` | Where to download / find the raw dataset |
| `--skip-download` | off | Skip the HuggingFace download (use existing files in `--raw-dir`) |

```bash
# Re-process without re-downloading
python dataset/construct_dataset.py --skip-download

# Custom paths
python dataset/construct_dataset.py --raw-dir /data/lekiwi --output ./data/dataset.npz
```

### Action discretization

Raw continuous actions are mapped to class indices:

| Component | < 0 | = 0 | > 0 |
|-----------|-----|-----|-----|
| x.vel | −0.3 (class 0) | 0.0 (class 1) | +0.3 (class 2) |
| y.vel | −0.3 (class 0) | 0.0 (class 1) | +0.3 (class 2) |
| θ.vel | −90° (class 0) | 0° (class 1) | +90° (class 2) |

---

## 2. Training (CNN)

Loads `data/dataset.npz`, splits into train/test, trains the CNN, evaluates on the held-out test set, and saves the model weights and a training curve plot.

```bash
python models/cnn_train.py
```

Output:
- `models/checkpoints/cnn_policy.pth` — best-validation-loss weights
- `models/checkpoints/training_curve.png` — loss and accuracy curves

**Options**

| Flag | Default | Description |
|------|---------|-------------|
| `--data PATH` | `./data/dataset.npz` | Path to the `.npz` dataset |
| `--epochs N` | `30` | Number of training epochs |
| `--batch-size N` | `64` | Mini-batch size |
| `--lr FLOAT` | `1e-3` | Adam learning rate |
| `--val-split FLOAT` | `0.1` | Fraction of training data used for validation |
| `--test-size FLOAT` | `0.15` | Fraction of total data held out for testing |
| `--output-dir PATH` | `./models/checkpoints` | Where to save weights and plots |
| `--device` | auto | `cuda` or `cpu` |

```bash
python models/cnn_train.py --epochs 50 --lr 3e-4 --device cuda
```

---

## 3. Inference (CNN)

Load a trained policy and get real-valued controls from a camera frame:

```python
import cv2
from models.cnn_policy import CNNPolicy

policy = CNNPolicy()
policy.load("models/checkpoints/cnn_policy.pth")

frame = cv2.cvtColor(cv2.imread("frame.png"), cv2.COLOR_BGR2RGB)
controls = policy.get_controls(frame)
# {'x_vel': 0.3, 'y_vel': 0.0, 'theta_vel': 0.0}
```

`get_controls()` accepts any uint8 RGB image — it resizes internally to match the training resolution.

---

## Project structure

```
behavior_cloning/
├── dataset/
│   └── construct_dataset.py   # download & preprocess
├── models/
│   ├── cnn_policy.py          # CNNPolicy class
│   ├── cnn_train.py           # training script
│   └── checkpoints/           # saved weights & plots (created at runtime)
├── data/
│   ├── raw/                   # downloaded HuggingFace files (created at runtime)
│   └── dataset.npz            # preprocessed dataset (created at runtime)
├── setup_venv.ps1
├── setup_venv.sh
├── setup_venv.zsh
└── requirements.txt
```
