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
    """Name portion of a CALFW/CPLFW filename, i.e. everything before the
    trailing "_<index>.jpg" (e.g. "Carl_Reiner_0001.jpg" -> "Carl_Reiner")."""
    return re.sub(r"_\d+\.(jpg|jpeg|png)$", "", filename, flags=re.IGNORECASE)


def parse_flat_pairs(pairs_path: Path):
    """
    CALFW/CPLFW format: two consecutive lines per pair, "filename number" each.

    The trailing number is NOT a match/non-match label:
      - CALFW: cross-validation fold index (1-10) for genuine pairs, 0 for impostors.
      - CPLFW: 1 for genuine pairs, 0 for impostors (looks like a label, but isn't
        one we should rely on across datasets).

    Match vs non-match is derived from the filenames instead: a pair is a match
    iff both images belong to the same identity (same name portion). The file is
    also expected to be ordered as all genuine pairs first, then all impostors;
    we cross-check that and warn (rather than silently trust it) if it breaks.
    """
    with open(pairs_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    pairs = []
    for i in range(0, len(lines) - 1, 2):
        file1 = lines[i].split()[0]
        file2 = lines[i + 1].split()[0]
        label = 1 if _identity_of(file1) == _identity_of(file2) else 0
        pairs.append((file1, file2, label))

    labels = [label for _, _, label in pairs]
    num_match = sum(labels)
    if num_match == 0 or num_match == len(labels):
        raise ValueError(
            f"{pairs_path}: derived all pairs as the same class "
            f"({num_match}/{len(labels)} matches); filename parsing is likely wrong."
        )
    # Expected layout: matches first, then non-matches. Warn if violated.
    first_nonmatch = next((j for j, v in enumerate(labels) if v == 0), len(labels))
    if any(labels[j] == 1 for j in range(first_nonmatch, len(labels))):
        print(f"  (warning: {pairs_path} is not ordered matches-then-nonmatches; "
              f"labels derived from filenames anyway)")
    return pairs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
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
    """FAR = wrongly-accepted non-matches. FRR = wrongly-rejected matches."""
    far_values, tpr_values, thresholds = roc_curve(labels, similarities)
    frr_values = 1 - tpr_values
    closest_index = np.argmin(np.abs(thresholds - threshold))
    return float(far_values[closest_index]), float(frr_values[closest_index])


def compute_accuracy(similarities: np.ndarray, labels: np.ndarray, threshold: float) -> float:
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