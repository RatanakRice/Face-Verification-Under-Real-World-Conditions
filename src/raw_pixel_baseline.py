"""
Raw-pixel baseline: the simplest possible "embedding" (no learning, no PCA),
scored through the exact same EER-threshold/accuracy/FAR/FRR pipeline as
ArcFace and face_recognition, for a direct, apples-to-apples comparison.

Embedding: greyscale -> resize to 32x32 (cv2.INTER_AREA) -> flatten -> L2
normalise. Threshold fit once at the EER on clean LFW, then applied
unchanged to CALFW and CPLFW (same convention as every other condition in
this project).
"""
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from compare import (
    compute_accuracy,
    compute_far_frr,
    cosine_similarity,
    find_eer_threshold,
    parse_flat_pairs,
    parse_lfw_pairs,
)

PAIRS_DIR = Path("data/pairs")
ALIGNED_DIR = Path("data2/processed/aligned")

DATASETS = [
    ("lfw_bigger", PAIRS_DIR / "pairs_LFW.txt", "lfw"),
    ("calfw_bigger", PAIRS_DIR / "pairs_CALFW.txt", "flat"),
    ("cplfw_bigger", PAIRS_DIR / "pairs_CPLFW.txt", "flat"),
]


def embed_raw_pixels(image_path: Path) -> np.ndarray:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    vec = resized.astype(np.float32).flatten()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def embed_folder(folder: Path) -> dict:
    embeddings = {}
    for path in tqdm(sorted(folder.glob("*.jpg")), desc=f"Embedding {folder.name}", unit="img"):
        embeddings[path.name] = embed_raw_pixels(path)
    return embeddings


def score(pairs, embeddings: dict, threshold: float = None):
    sims, labels = [], []
    for file1, file2, label in pairs:
        if file1 in embeddings and file2 in embeddings:
            sims.append(cosine_similarity(embeddings[file1], embeddings[file2]))
            labels.append(label)
    sims = np.array(sims)
    labels = np.array(labels)
    if threshold is None:
        threshold = find_eer_threshold(sims, labels)
    accuracy = compute_accuracy(sims, labels, threshold)
    far, frr = compute_far_frr(sims, labels, threshold)
    return threshold, accuracy, far, frr, len(sims)


def main():
    t0 = time.time()
    threshold = None
    for name, pairs_path, fmt in DATASETS:
        parser_fn = parse_lfw_pairs if fmt == "lfw" else parse_flat_pairs
        pairs = parser_fn(pairs_path)
        embeddings = embed_folder(ALIGNED_DIR / name)

        is_baseline = threshold is None
        th, accuracy, far, frr, n = score(pairs, embeddings, threshold=threshold)
        if is_baseline:
            threshold = th

        print(f"{name:15s} threshold={th:.4f}  accuracy={accuracy*100:.2f}%  "
              f"FAR={far*100:.2f}%  FRR={frr*100:.2f}%  pairs_scored={n}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

# Usage: PYTHONPATH=src python3 src/raw_pixel_baseline.py
