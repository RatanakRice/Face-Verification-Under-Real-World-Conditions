"""
Multi-seed variance check for the bonus quality-gate's realistic
"mixed-collection" result (Project Plan, Phase 7 / Report Section 5.7).

The reported mixdata result used one filename-seeded random assignment of
conditions to images. That's reproducible, but it's a single draw -- there
is no variance estimate for "would the gate's reported gain hold under a
different random mix?" This script re-draws the per-image condition
assignment with several different seed prefixes (same CONDITIONS/weights as
build_mixdata.py) and reports mean +/- std of the gate's accuracy delta
across seeds, for lfw_bigger, both models.

Runs fully in memory: for each seed, each image's chosen-condition version
is read once, embedded (unfixed), run through quality_gate.assess(), and
the fixed version embedded too -- no per-seed image folders or embedding
JSON files are written to disk, since only the aggregate statistic is
needed, not a reusable artifact. Models are loaded once and reused across
all seeds.
"""
import argparse
import json
import random
import statistics
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from build_mixdata import CONDITIONS, source_paths
from compare import cosine_similarity, find_eer_threshold, load_embeddings, parse_lfw_pairs
from embed import embed_arcface, embed_face_recognition, load_arcface_model
from mcnemar_test import mcnemar_table, mcnemar_test
from quality_gate import assess

PAIRS_PATH = Path("data/pairs/pairs_LFW.txt")
BASELINE_EMBEDDINGS_DIR = Path("data2/embeddings/aligned/lfw_bigger")
PAIRS_FILE = Path("model/mesh_pairs.npy")


def baseline_threshold(model: str) -> float:
    """Same EER-on-clean-LFW threshold used everywhere else in the project."""
    pairs = parse_lfw_pairs(PAIRS_PATH)
    embeddings = load_embeddings(BASELINE_EMBEDDINGS_DIR / f"{model}.json")
    sims, labels = [], []
    for file1, file2, label in pairs:
        if file1 in embeddings and file2 in embeddings:
            sims.append(cosine_similarity(embeddings[file1], embeddings[file2]))
            labels.append(label)
    return find_eer_threshold(np.array(sims), np.array(labels))


def run_one_seed(seed_prefix: str, image_names: list, sources: dict, arcface_model, pairs_table) -> dict:
    """
    Build one random condition mix (in memory), run every image through the
    quality gate, and return per-image embeddings for the unfixed and
    gate-fixed versions, both models.
    """
    unfixed = {"arcface": {}, "face_recognition": {}}
    fixed = {"arcface": {}, "face_recognition": {}}
    rejected = 0

    for name in tqdm(image_names, desc=f"seed={seed_prefix}", unit="img"):
        rng = random.Random(seed_prefix + name)
        condition = rng.choice(CONDITIONS)
        src = sources[condition] / name
        image = cv2.imread(str(src))
        if image is None:
            continue

        unfixed["arcface"][name] = embed_arcface(arcface_model, image)
        fr_vec = embed_face_recognition(image)
        if fr_vec is not None:
            unfixed["face_recognition"][name] = fr_vec

        gated_image, info = assess(image, pairs_table)
        if info["rejected"]:
            rejected += 1
            continue

        fixed["arcface"][name] = embed_arcface(arcface_model, gated_image)
        fr_vec_fixed = embed_face_recognition(gated_image)
        if fr_vec_fixed is not None:
            fixed["face_recognition"][name] = fr_vec_fixed

    return {"unfixed": unfixed, "fixed": fixed, "rejected": rejected, "total": len(image_names)}


def score(pairs, embeddings: dict, threshold: float) -> dict:
    """Per-pair correctness (compare.py convention), as a dict keyed by pair."""
    correctness = {}
    for file1, file2, label in pairs:
        if file1 not in embeddings or file2 not in embeddings:
            continue
        sim = cosine_similarity(embeddings[file1], embeddings[file2])
        correctness[(file1, file2)] = (sim >= threshold) == bool(label)
    return correctness


def main():
    parser = argparse.ArgumentParser(description="Multi-seed variance check for the mixdata quality-gate result (lfw_bigger).")
    parser.add_argument("--processed-dir", default="data2/processed")
    parser.add_argument("--dataset", default="lfw_bigger")
    parser.add_argument("--seeds", nargs="+", default=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"])
    parser.add_argument("--output", default="data2/processed/mixdata_variance.json")
    args = parser.parse_args()

    sources = source_paths(Path(args.processed_dir), args.dataset)
    image_names = sorted(p.name for p in sources["clean"].glob("*.jpg"))
    print(f"{len(image_names)} images in {args.dataset}, {len(args.seeds)} seeds: {args.seeds}")

    pairs_table = np.load(PAIRS_FILE)
    pairs = parse_lfw_pairs(PAIRS_PATH)
    arcface_model = load_arcface_model()

    thresholds = {"arcface": baseline_threshold("arcface"), "face_recognition": baseline_threshold("face_recognition")}
    print(f"Thresholds: {thresholds}")

    per_seed_results = []
    for seed_prefix in args.seeds:
        run = run_one_seed(seed_prefix, image_names, sources, arcface_model, pairs_table)
        coverage = 1 - run["rejected"] / run["total"]

        row = {"seed": seed_prefix, "coverage": coverage}
        for model in ["arcface", "face_recognition"]:
            correctness_before = score(pairs, run["unfixed"][model], thresholds[model])
            correctness_after = score(pairs, run["fixed"][model], thresholds[model])
            table = mcnemar_table(correctness_before, correctness_after)
            test = mcnemar_test(table["b_worse"], table["c_better"])
            n = table["n_common"]
            acc_before = (table["both_right"] + table["b_worse"]) / n if n else float("nan")
            acc_after = (table["both_right"] + table["c_better"]) / n if n else float("nan")
            row[model] = {
                "n_pairs": n,
                "accuracy_unfixed": acc_before,
                "accuracy_gated": acc_after,
                "delta_pts": (acc_after - acc_before) * 100,
                "p_exact": test["p_exact"],
            }
        per_seed_results.append(row)
        print(f"  seed={seed_prefix}  coverage={coverage:.3f}  "
              f"arcface_delta={row['arcface']['delta_pts']:+.2f}pt (p={row['arcface']['p_exact']:.2e})  "
              f"face_recognition_delta={row['face_recognition']['delta_pts']:+.2f}pt (p={row['face_recognition']['p_exact']:.2e})")

    summary = {}
    for model in ["arcface", "face_recognition"]:
        deltas = [r[model]["delta_pts"] for r in per_seed_results]
        summary[model] = {
            "deltas_pts": deltas,
            "mean_delta_pts": statistics.mean(deltas),
            "std_delta_pts": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "min_delta_pts": min(deltas),
            "max_delta_pts": max(deltas),
        }
    coverages = [r["coverage"] for r in per_seed_results]
    summary["coverage"] = {
        "values": coverages,
        "mean": statistics.mean(coverages),
        "std": statistics.stdev(coverages) if len(coverages) > 1 else 0.0,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"per_seed": per_seed_results, "summary": summary}, f, indent=2)

    print(f"\n=== Summary across {len(args.seeds)} seeds ===")
    for model in ["arcface", "face_recognition"]:
        s = summary[model]
        print(f"{model:20s} mean={s['mean_delta_pts']:+.3f}pt  std={s['std_delta_pts']:.3f}pt  "
              f"range=[{s['min_delta_pts']:+.2f}, {s['max_delta_pts']:+.2f}]pt")
    print(f"coverage            mean={summary['coverage']['mean']:.3f}  std={summary['coverage']['std']:.4f}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

# Usage: PYTHONPATH=src python3 src/mixdata_variance.py --dataset lfw_bigger --seeds v1 v2 v3 v4 v5 v6 v7 v8
