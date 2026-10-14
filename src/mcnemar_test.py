"""
Statistical significance testing for every before/after correction comparison
in the report: lighting (auto-gamma), occlusion (mirror-fill), and the
bonus quality-gate mix test.

Every comparison is the *same pairs*, scored twice (once degraded/unfixed,
once corrected/fixed) -- a paired, binary-outcome setup (each pair is either
classified correctly or not). McNemar's test is the textbook-correct choice
here, not a t-test: it ignores pairs whose classification didn't change
between the two runs and tests only whether the pairs that *did* flip are
split evenly between "got better" and "got worse", or whether one direction
dominates.

No new images or embeddings are generated -- this reads the same embedding
JSON files and pairs files run_scorecard_bigger.py already uses, and reuses
compare.py's own loading/parsing/similarity functions so the per-pair
correctness here is computed exactly the same way the existing scorecards
were.
"""
import argparse
import csv
from pathlib import Path

from scipy.stats import binomtest, chi2

from compare import (
    cosine_similarity,
    find_eer_threshold,
    load_embeddings,
    parse_flat_pairs,
    parse_lfw_pairs,
)

PAIRS_DIR = Path("data/pairs")
EMBEDDINGS_DIR = Path("data2/embeddings")

DATASETS = ["lfw_bigger", "calfw_bigger", "cplfw_bigger"]
GAMMA_VALUES = [0.4, 1.8, 2.5]
MODELS = ["arcface", "face_recognition"]

PAIRS_BY_FAMILY = {
    "lfw": (PAIRS_DIR / "pairs_LFW.txt", "lfw"),
    "calfw": (PAIRS_DIR / "pairs_CALFW.txt", "flat"),
    "cplfw": (PAIRS_DIR / "pairs_CPLFW.txt", "flat"),
}


def family_of(dataset: str) -> str:
    return dataset.removesuffix("_bigger")


def per_pair_correctness(pairs, embeddings: dict, threshold: float) -> dict:
    """
    {(file1, file2): bool correct}, restricted to pairs with an embedding
    for both images -- same restriction compare.py's compute_similarities
    applies, just kept per-pair instead of collapsed into one accuracy number.
    """
    correctness = {}
    for file1, file2, label in pairs:
        if file1 not in embeddings or file2 not in embeddings:
            continue
        sim = cosine_similarity(embeddings[file1], embeddings[file2])
        predicted_match = sim >= threshold
        correctness[(file1, file2)] = predicted_match == bool(label)
    return correctness


def mcnemar_table(correctness_before: dict, correctness_after: dict) -> dict:
    """
    Restrict to pairs scorable under *both* conditions (the "fair",
    matched-subset convention used throughout the project whenever a
    correction can reject/drop images), then build the 2x2 discordant-pair
    counts McNemar's test needs.

    b_worse  = correct before, incorrect after  (correction hurt this pair)
    c_better = incorrect before, correct after  (correction helped this pair)
    """
    common = correctness_before.keys() & correctness_after.keys()
    both_right = both_wrong = b_worse = c_better = 0

    for key in common:
        before, after = correctness_before[key], correctness_after[key]
        if before and after:
            both_right += 1
        elif not before and not after:
            both_wrong += 1
        elif before and not after:
            b_worse += 1
        else:
            c_better += 1

    return {
        "n_common": len(common),
        "both_right": both_right,
        "both_wrong": both_wrong,
        "b_worse": b_worse,
        "c_better": c_better,
    }


def mcnemar_test(b_worse: int, c_better: int):
    """
    Exact two-sided McNemar's test (binomial on the discordant pairs, always
    valid regardless of sample size) as the primary result, plus the
    continuity-corrected chi-square statistic for reference/reporting
    alongside the p-value.
    """
    n = b_worse + c_better
    if n == 0:
        return {"n_discordant": 0, "chi2": 0.0, "p_chi2": 1.0, "p_exact": 1.0}

    chi2_stat = (abs(b_worse - c_better) - 1) ** 2 / n
    p_chi2 = float(1 - chi2.cdf(chi2_stat, df=1))
    p_exact = float(binomtest(min(b_worse, c_better), n, 0.5, alternative="two-sided").pvalue)

    return {"n_discordant": n, "chi2": float(chi2_stat), "p_chi2": p_chi2, "p_exact": p_exact}


def run_comparison(name: str, pairs_path: Path, pairs_format: str, embeddings_before_path: Path,
                    embeddings_after_path: Path, threshold: float) -> dict:
    parser_fn = parse_lfw_pairs if pairs_format == "lfw" else parse_flat_pairs
    pairs = parser_fn(pairs_path)

    embeddings_before = load_embeddings(embeddings_before_path)
    embeddings_after = load_embeddings(embeddings_after_path)

    correctness_before = per_pair_correctness(pairs, embeddings_before, threshold)
    correctness_after = per_pair_correctness(pairs, embeddings_after, threshold)

    table = mcnemar_table(correctness_before, correctness_after)
    test = mcnemar_test(table["b_worse"], table["c_better"])

    n = table["n_common"]
    accuracy_before = (table["both_right"] + table["b_worse"]) / n if n else float("nan")
    accuracy_after = (table["both_right"] + table["c_better"]) / n if n else float("nan")

    return {
        "comparison": name,
        "n_pairs": n,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "delta_pts": (accuracy_after - accuracy_before) * 100,
        **table,
        **test,
    }


def build_comparisons(embeddings_dir: Path) -> list:
    """Every before/after cell reported in the Results section: lighting
    correction, occlusion correction, and the bonus quality-gate mix test."""
    comparisons = []

    for model in MODELS:
        baseline_pairs_path, baseline_pairs_format = PAIRS_BY_FAMILY["lfw"]
        baseline_embeddings_path = embeddings_dir / "aligned" / "lfw_bigger" / f"{model}.json"
        baseline_pairs = parse_lfw_pairs(baseline_pairs_path)
        baseline_embeddings = load_embeddings(baseline_embeddings_path)
        _, baseline_labels = [], []
        similarities, labels = [], []
        for file1, file2, label in baseline_pairs:
            if file1 in baseline_embeddings and file2 in baseline_embeddings:
                similarities.append(cosine_similarity(baseline_embeddings[file1], baseline_embeddings[file2]))
                labels.append(label)
        import numpy as np
        threshold = find_eer_threshold(np.array(similarities), np.array(labels))

        for dataset in DATASETS:
            pairs_path, pairs_format = PAIRS_BY_FAMILY[family_of(dataset)]

            for gamma in GAMMA_VALUES:
                comparisons.append(dict(
                    name=f"lighting | {model} | {dataset} | gamma={gamma}",
                    pairs_path=pairs_path, pairs_format=pairs_format,
                    embeddings_before_path=embeddings_dir / "lighting" / dataset / f"gamma_{gamma}" / f"{model}.json",
                    embeddings_after_path=embeddings_dir / "lighting_corrected" / dataset / f"gamma_{gamma}" / f"{model}.json",
                    threshold=threshold,
                ))

            comparisons.append(dict(
                name=f"occlusion | {model} | {dataset} | block_40_40",
                pairs_path=pairs_path, pairs_format=pairs_format,
                embeddings_before_path=embeddings_dir / "occlusion" / dataset / "block_40_40" / f"{model}.json",
                embeddings_after_path=embeddings_dir / "occlusion_fixed" / dataset / "block_40_40" / f"{model}.json",
                threshold=threshold,
            ))

            comparisons.append(dict(
                name=f"mixdata_gate | {model} | {dataset}",
                pairs_path=pairs_path, pairs_format=pairs_format,
                embeddings_before_path=embeddings_dir / f"mixdata_{dataset}" / f"{model}.json",
                embeddings_after_path=embeddings_dir / "quality_gate" / f"mixdata_{dataset}" / f"{model}.json",
                threshold=threshold,
            ))

    return comparisons


def main():
    parser = argparse.ArgumentParser(description="McNemar's test on every before/after correction comparison.")
    parser.add_argument("--embeddings-dir", default=str(EMBEDDINGS_DIR))
    parser.add_argument("--output", default="data2/processed/mcnemar_results.csv")
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    comparisons = build_comparisons(embeddings_dir)
    alpha = 0.05
    alpha_bonferroni = alpha / len(comparisons)

    results = []
    for spec in comparisons:
        name = spec.pop("name")
        print(f"Scoring: {name}")
        result = run_comparison(name, **spec)
        results.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{len(comparisons)} comparisons run. Bonferroni-adjusted alpha = {alpha_bonferroni:.6f} "
          f"(family-wise alpha {alpha} / {len(comparisons)} comparisons)\n")
    print(f"{'comparison':45s} {'n':>6s} {'before':>8s} {'after':>8s} {'delta':>7s} "
          f"{'b':>5s} {'c':>5s} {'p_exact':>10s} {'sig?':>6s}")
    for r in results:
        sig = "**" if r["p_exact"] < alpha_bonferroni else ("*" if r["p_exact"] < alpha else "")
        print(f"{r['comparison']:45s} {r['n_pairs']:6d} {r['accuracy_before']*100:7.2f}% {r['accuracy_after']*100:7.2f}% "
              f"{r['delta_pts']:+6.2f} {r['b_worse']:5d} {r['c_better']:5d} {r['p_exact']:10.2e} {sig:>6s}")
    print("\n* p < 0.05 (uncorrected)   ** p < Bonferroni-adjusted alpha")
    print(f"Saved full results to {output_path}")


if __name__ == "__main__":
    main()

# Usage: PYTHONPATH=src python3 src/mcnemar_test.py --embeddings-dir data2/embeddings --output data2/processed/mcnemar_results.csv
