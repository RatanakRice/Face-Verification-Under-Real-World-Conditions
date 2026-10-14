"""
Diagnose a low clean-LFW baseline.

1. Prints genuine vs impostor similarity stats (a healthy ArcFace run has
   genuine pairs around 0.5-0.7 and impostor pairs around 0.0-0.1).
2. Saves a contact sheet of the 24 worst genuine pairs (same person, lowest
   similarity). If these show the wrong person, a bystander, or a skewed or
   cut-off crop, the bug is in detect_align.py, not in the models.

Usage:
python src/diagnose_baseline.py --pairs data/pairs/pairs_LFW.txt \
    --embeddings data/processed/embeddings/lfw_arcface.json \
    --aligned data/processed/aligned/lfw --out worst_genuine_pairs.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from compare import cosine_similarity, load_embeddings, parse_lfw_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--aligned", required=True, help="Folder of aligned crops")
    parser.add_argument("--out", default="worst_genuine_pairs.png")
    parser.add_argument("--n", type=int, default=24)
    args = parser.parse_args()

    pairs = parse_lfw_pairs(Path(args.pairs))
    emb = load_embeddings(Path(args.embeddings))

    genuine, impostor = [], []
    for f1, f2, label in pairs:
        if f1 in emb and f2 in emb:
            sim = cosine_similarity(emb[f1], emb[f2])
            (genuine if label else impostor).append((sim, f1, f2))

    g = np.array([s for s, _, _ in genuine])
    i = np.array([s for s, _, _ in impostor])
    print(f"Genuine  mean {g.mean():.3f}  median {np.median(g):.3f}")
    print(f"Impostor mean {i.mean():.3f}  median {np.median(i):.3f}")
    print(f"Genuine pairs below 0.2: {np.mean(g < 0.2):.1%}")
    print(f"Impostor pairs above 0.2: {np.mean(i > 0.2):.1%}")

    rows = []
    for sim, f1, f2 in sorted(genuine)[: args.n]:
        a = cv2.resize(cv2.imread(str(Path(args.aligned) / f1)), (160, 160))
        b = cv2.resize(cv2.imread(str(Path(args.aligned) / f2)), (160, 160))
        row = np.hstack([a, b])
        cv2.putText(row, f"{sim:.2f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        rows.append(row)

    cols = 4
    while len(rows) % cols:
        rows.append(np.zeros_like(rows[0]))
    grid = np.vstack([np.hstack(rows[k:k + cols]) for k in range(0, len(rows), cols)])
    cv2.imwrite(args.out, grid)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()