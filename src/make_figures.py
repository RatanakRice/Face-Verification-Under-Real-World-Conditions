"""
Regenerates Fig. 2 (8-seed gate accuracy-change plot) from
data2/processed/mixdata_variance.json.

Report section: Results and Analysis / Run Distributions.
Input: data2/processed/mixdata_variance.json (written by mixdata_variance.py).
Output: figures/seed_variance.pdf.

Note: Fig. 1 (the pipeline schematic) is a hand-drawn diagram, not generated
from data, so there is no script for it here.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VARIANCE_PATH = Path("data2/processed/mixdata_variance.json")
OUTPUT_PATH = Path("figures/seed_variance.pdf")


def main():
    with open(VARIANCE_PATH) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    models = [("arcface", "ArcFace"), ("face_recognition", "face_recognition")]
    for i, (key, label) in enumerate(models):
        deltas = np.array(data["summary"][key]["deltas_pts"])
        mean = data["summary"][key]["mean_delta_pts"]
        std = data["summary"][key]["std_delta_pts"]
        xs = np.full(len(deltas), i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(deltas))
        ax.scatter(xs, deltas, s=14, alpha=0.7, label=label)
        ax.errorbar([i], [mean], yerr=[std], fmt="_", color="black", capsize=4, markersize=14)

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([m[1] for m in models], fontsize=8)
    ax.set_ylabel("Accuracy change (pt)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
