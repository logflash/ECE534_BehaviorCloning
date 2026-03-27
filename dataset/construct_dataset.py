"""
construct_dataset.py

Downloads the TagAggDann/lekiwi_green_block dataset from HuggingFace,
decodes video frames, discretizes actions, and saves a .npz dataset.

Usage:
    python dataset/construct_dataset.py
    python dataset/construct_dataset.py --output ./data/dataset.npz --raw-dir ./data/raw
"""

import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download


REPO_ID = "TagAggDann/lekiwi_green_block"
IMAGE_SIZE = 84  # resize frames to IMAGE_SIZE x IMAGE_SIZE


def discretize_action(action: list[float]) -> list[int]:
    """
    Convert a raw [x.vel, y.vel, theta.vel] action to class indices.

    x.vel / y.vel: {<0 → 0 (-0.3), ==0 → 1 (0.0), >0 → 2 (+0.3)}
    theta.vel:     {<0 → 0 (-90),  ==0 → 1 (0),    >0 → 2 (+90)}
    """
    def cls_xy(v: float) -> int:
        if abs(v) < 1e-6:
            return 1
        return 0 if v < 0 else 2

    def cls_theta(v: float) -> int:
        if abs(v) < 1e-6:
            return 1
        return 0 if v < 0 else 2

    return [cls_xy(action[0]), cls_xy(action[1]), cls_theta(action[2])]


def load_parquet_labels(raw_dir: str) -> pd.DataFrame:
    """Read all episode parquet files and return a DataFrame sorted by global index."""
    pattern = os.path.join(raw_dir, "data", "chunk-*", "file-*.parquet")
    parquet_files = sorted(glob.glob(pattern))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found matching {pattern}. "
            "Did the download complete successfully?"
        )

    dfs = []
    for path in parquet_files:
        df = pd.read_parquet(path, columns=["index", "episode_index", "frame_index", "action"])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("index").reset_index(drop=True)
    return df


def decode_video(video_path: str) -> list[np.ndarray]:
    """
    Decode all frames from an MP4 as uint8 RGB arrays resized to IMAGE_SIZE x IMAGE_SIZE.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()
    return frames


def build_dataset(raw_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Build image and label arrays from the downloaded dataset.

    The dataset stores all episodes as a single concatenated video per chunk.
    The global `index` column in the parquet maps each row to its frame position
    in that concatenated video.

    Returns:
        images: np.ndarray of shape (N, IMAGE_SIZE, IMAGE_SIZE, 3), dtype uint8
        labels: np.ndarray of shape (N, 3), dtype int64
                columns are class indices for [x.vel, y.vel, theta.vel]
    """
    df = load_parquet_labels(raw_dir)
    N = len(df)

    # Decode all video files in the chunk(s), concatenating in sorted order.
    # Each chunk's frames are stored in one or more MP4 files named file-000.mp4, etc.
    video_pattern = os.path.join(
        raw_dir, "videos", "observation.images.front", "chunk-*", "file-*.mp4"
    )
    video_files = sorted(glob.glob(video_pattern))
    if not video_files:
        raise FileNotFoundError(
            f"No video files found matching {video_pattern}. "
            "Did the download complete successfully?"
        )

    all_frames: list[np.ndarray] = []
    for vf in video_files:
        print(f"  decoding {vf} ...")
        all_frames.extend(decode_video(vf))

    n_frames = len(all_frames)
    print(f"  total frames decoded: {n_frames}  |  parquet rows: {N}")

    if n_frames != N:
        print(f"  [warn] frame/row count mismatch — trimming to minimum")
        n = min(n_frames, N)
        all_frames = all_frames[:n]
        df = df.iloc[:n]

    # The global `index` column is the frame position in the concatenated video.
    # After sorting by `index` the rows are already aligned with all_frames positionally.
    frame_indices = df["index"].to_numpy()
    max_idx = frame_indices.max()
    if max_idx >= len(all_frames):
        raise ValueError(
            f"Parquet references frame index {max_idx} but only {len(all_frames)} frames decoded."
        )

    images = np.stack([all_frames[i] for i in frame_indices], axis=0)  # (N, H, W, 3)
    labels = np.array(
        [discretize_action(a) for a in df["action"].tolist()],
        dtype=np.int64,
    )  # (N, 3)

    # Print per-episode summary
    for ep_idx, ep_df in df.groupby("episode_index"):
        print(f"  episode {ep_idx:3d}: {len(ep_df)} frames")

    assert images.shape[0] == labels.shape[0], "Frame/label count mismatch after assembly"
    return images, labels


def main():
    parser = argparse.ArgumentParser(description="Construct behavior cloning dataset")
    parser.add_argument(
        "--output", default="./data/dataset.npz",
        help="Path to save the output .npz file (default: ./data/dataset.npz)"
    )
    parser.add_argument(
        "--raw-dir", default="./data/raw",
        help="Directory to download/find the raw HuggingFace dataset (default: ./data/raw)"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloading and use existing files in --raw-dir"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Step 1: Download dataset
    if not args.skip_download:
        print(f"Downloading {REPO_ID} to {args.raw_dir} ...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=args.raw_dir,
        )
        print("Download complete.")
    else:
        print(f"Skipping download; using existing files in {args.raw_dir}")

    # Step 2: Build arrays
    print("\nProcessing episodes ...")
    images, labels = build_dataset(args.raw_dir)

    print(f"\nDataset assembled:")
    print(f"  images: {images.shape} {images.dtype}")
    print(f"  labels: {labels.shape} {labels.dtype}")

    # Step 3: Save
    print(f"\nSaving to {args.output} ...")
    np.savez_compressed(args.output, images=images, labels=labels)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved ({size_mb:.1f} MB).")

    # Label distribution summary
    names = ["x.vel", "y.vel", "theta.vel"]
    class_names = {0: "neg", 1: "zero", 2: "pos"}
    print("\nLabel distribution:")
    for i, name in enumerate(names):
        counts = np.bincount(labels[:, i], minlength=3)
        total = counts.sum()
        dist = {class_names[c]: f"{counts[c]} ({100*counts[c]/total:.1f}%)" for c in range(3)}
        print(f"  {name}: {dist}")


if __name__ == "__main__":
    main()
