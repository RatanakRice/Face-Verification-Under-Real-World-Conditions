"""
Scores the lighting/occlusion pilot conditions from data/processed/embeddings/
(the earlier, smaller pilot run; mild occlusion sizes, superseded by the 40%
box used in the final results).

Input: data/processed/embeddings/.
Output: data/processed/scorecard_lighting_occlusion.txt.
Report section: pilot work behind Parameter Settings.
"""
from datetime import datetime
from pathlib import Path

from compare import score_condition

EMBEDDINGS_DIR = Path("data/processed/embeddings")
PAIRS_DIR = Path("data/pairs")
OUTPUT_PATH = Path("data/processed/scorecard_lighting_occlusion.txt")

DATASETS = ["lfw", "calfw", "cplfw", "lfw_bigger", "calfw_bigger", "cplfw_bigger"]
from constants import GAMMA_VALUES
OCCLUSION_CONFIGS = [("random", 0.25), ("random", 0.5)]
MODELS = ["arcface", "face_recognition"]

# (pairs file, pairs format) per dataset family, keyed by the family name
# with any "_bigger" suffix stripped.
PAIRS_BY_FAMILY = {
    "lfw": (PAIRS_DIR / "pairs_LFW.txt", "lfw"),
    "calfw": (PAIRS_DIR / "pairs_CALFW.txt", "flat"),
    "cplfw": (PAIRS_DIR / "pairs_CPLFW.txt", "flat"),
}


def family_of(dataset: str) -> str:
    return dataset.removesuffix("_bigger")


def run_all():
    lines = [f"Lighting/occlusion scorecard generated {datetime.now().isoformat(timespec='seconds')}", ""]

    for model in MODELS:
        lines.append(f"=== {model} ===")

        # Baseline threshold comes from the clean LFW pairs, same fixed
        # threshold used for the core baseline/CALFW/CPLFW scorecard.
        baseline_pairs_path, baseline_pairs_format = PAIRS_BY_FAMILY["lfw"]
        baseline_embeddings_path = EMBEDDINGS_DIR / f"lfw_{model}.json"
        baseline_result = score_condition(baseline_pairs_path, baseline_pairs_format, baseline_embeddings_path, threshold=None)
        threshold = baseline_result["threshold"]
        lines.append(f"(threshold from LFW baseline EER: {threshold:.4f})")

        for dataset in DATASETS:
            pairs_path, pairs_format = PAIRS_BY_FAMILY[family_of(dataset)]

            for gamma in GAMMA_VALUES:
                label = f"{dataset} lighting gamma={gamma}"
                embeddings_path = EMBEDDINGS_DIR / "lighting" / dataset / f"gamma_{gamma}" / f"{model}.json"
                result = score_condition(pairs_path, pairs_format, embeddings_path, threshold)
                lines.append(
                    f"{label:35s} accuracy={result['accuracy']:.4f}  "
                    f"FAR={result['far']:.4f}  FRR={result['frr']:.4f}  "
                    f"pairs_scored={result['num_pairs_scored']}"
                )

            for side, mask_fraction in OCCLUSION_CONFIGS:
                label = f"{dataset} occlusion {side}_{mask_fraction}"
                embeddings_path = EMBEDDINGS_DIR / "occlusion" / f"{dataset}_{side}_{mask_fraction}_{model}.json"
                result = score_condition(pairs_path, pairs_format, embeddings_path, threshold)
                lines.append(
                    f"{label:35s} accuracy={result['accuracy']:.4f}  "
                    f"FAR={result['far']:.4f}  FRR={result['frr']:.4f}  "
                    f"pairs_scored={result['num_pairs_scored']}"
                )

        lines.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_all()

# Usage: PYTHONPATH=src python3 src/run_script/run_scorecard_lighting_occlusion.py
