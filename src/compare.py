import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_curve


def load_embeddings(embeddings_path: Path) -> dict:
    """Load {filename: embedding_vector} from a JSON file saved by embed.py."""
    with open(embeddings_path) as f:
        raw = json.load(f)
    return {name: np.array(vector) for name, vector in raw.items()}


def parse_lfw_pairs(pairs_path: Path):
    """
    LFW format. First line is header, skipped.
    Match:    name  idx1  idx2
    Mismatch: name1 idx1  name2 idx2
    """
    pairs = []
    with open(pairs_path) as f:
        lines = f.readlines()[1:]

    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            name, idx1, idx2 = parts
            pairs.append((f"{name}_{int(idx1):04d}.jpg", f"{name}_{int(idx2):04d}.jpg", 1))
        elif len(parts) == 4:
            name1, idx1, name2, idx2 = parts
            pairs.append((f"{name1}_{int(idx1):04d}.jpg", f"{name2}_{int(idx2):04d}.jpg", 0))

    return pairs


def _identity_of(filename: str) -> str:
    """Name part of a filename: "Carl_Reiner_0001.jpg" -> "Carl_Reiner"."""
    return re.sub(r"_\d+\.(jpg|jpeg|png)$", "", filename, flags=re.IGNORECASE)


def parse_flat_pairs(pairs_path: Path):
    """
    CALFW/CPLFW format: two consecutive lines per pair, "filename number" each.

    The trailing number is a cross-validation fold index, not a label, so the
    match label is taken from the filenames: a pair matches when both images
    belong to the same identity.
    """
    with open(pairs_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    pairs = []
    for i in range(0, len(lines) - 1, 2):
        file1 = lines[i].split()[0]
        file2 = lines[i + 1].split()[0]
        label = 1 if _identity_of(file1) == _identity_of(file2) else 0
        pairs.append((file1, file2, label))

    # Sanity check: a real pairs file has both classes. All-one-class means the
    # filename parsing broke, and roc_curve would fail further down anyway.
    num_match = sum(label for _, _, label in pairs)
    if num_match == 0 or num_match == len(pairs):
        raise ValueError(f"{pairs_path}: every pair parsed as the same class; check the filename format.")
    return pairs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_similarities(pairs, embeddings: dict):
    """Cosine similarity per pair. Skips pairs missing an embedding."""
    similarities, labels, skipped = [], [], 0

    for file1, file2, label in pairs:
        if file1 not in embeddings or file2 not in embeddings:
            skipped += 1
            continue
        similarities.append(cosine_similarity(embeddings[file1], embeddings[file2]))
        labels.append(label)

    if skipped:
        print(f"  ({skipped} pairs skipped, missing embedding for one or both images)")

    return np.array(similarities), np.array(labels)


def find_eer_threshold(similarities: np.ndarray, labels: np.ndarray) -> float:
    """Equal Error Rate threshold: where FAR and FRR are closest to equal."""
    far_values, tpr_values, thresholds = roc_curve(labels, similarities)
    frr_values = 1 - tpr_values
    eer_index = np.nanargmin(np.abs(far_values - frr_values))
    return float(thresholds[eer_index])


def compute_far_frr(similarities: np.ndarray, labels: np.ndarray, threshold: float):
    """
    FAR = fraction of non-match pairs wrongly accepted (similarity >= threshold).
    FRR = fraction of match pairs wrongly rejected (similarity < threshold).

    Counted directly at the given threshold, not read off ROC sample points.
    """
    non_match_sims = similarities[labels == 0]
    match_sims = similarities[labels == 1]
    far = float(np.mean(non_match_sims >= threshold))
    frr = float(np.mean(match_sims < threshold))
    return far, frr


def compute_accuracy(similarities: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Fraction of pairs classified correctly at this threshold."""
    predictions = similarities >= threshold
    return float(accuracy_score(labels, predictions))


def score_condition(pairs_path: Path, pairs_format: str, embeddings_path: Path, threshold: float = None):
    """
    Score one condition. Pass threshold=None only for the baseline
    (finds its own EER threshold); pass a fixed value for every other condition.
    """
    parser_fn = parse_lfw_pairs if pairs_format == "lfw" else parse_flat_pairs
    pairs = parser_fn(pairs_path)

    embeddings = load_embeddings(embeddings_path)
    similarities, labels = compute_similarities(pairs, embeddings)

    if threshold is None:
        threshold = find_eer_threshold(similarities, labels)

    accuracy = compute_accuracy(similarities, labels, threshold)
    far, frr = compute_far_frr(similarities, labels, threshold)

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "far": far,
        "frr": frr,
        "num_pairs_scored": len(similarities),
    }


def main():
    parser = argparse.ArgumentParser(description="Score a condition: cosine similarity, threshold, accuracy/FAR/FRR.")
    parser.add_argument("--pairs", required=True, help="Path to the pairs file")
    parser.add_argument("--pairs-format", required=True, choices=["lfw", "flat"])
    parser.add_argument("--embeddings", required=True, help="Path to the embeddings JSON")
    parser.add_argument("--threshold", type=float, default=None, help="Omit only for the baseline run")
    args = parser.parse_args()

    result = score_condition(Path(args.pairs), args.pairs_format, Path(args.embeddings), args.threshold)

    print(f"Threshold used: {result['threshold']:.4f}")
    print(f"Accuracy:       {result['accuracy']:.4f}")
    print(f"FAR:            {result['far']:.4f}")
    print(f"FRR:            {result['frr']:.4f}")
    print(f"Pairs scored:   {result['num_pairs_scored']}")


if __name__ == "__main__":
    main()

# run: python src/compare.py --pairs data/pairs/pairs_LFW.txt --pairs-format lfw --embeddings data/processed/embeddings/lfw_arcface.json
