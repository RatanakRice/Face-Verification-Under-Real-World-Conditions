from datetime import datetime
from pathlib import Path

from compare import score_condition

EMBEDDINGS_DIR = Path("data/processed/embeddings")
PAIRS_DIR = Path("data/pairs")
OUTPUT_PATH = Path("data/processed/scorecard.txt")

# (label, pairs file, pairs format, embeddings file prefix)
CONDITIONS = [
    ("LFW (baseline)", PAIRS_DIR / "pairs_LFW.txt", "lfw", "lfw"),
    ("CALFW (age gap)", PAIRS_DIR / "pairs_CALFW.txt", "flat", "calfw"),
    ("CPLFW (pose)", PAIRS_DIR / "pairs_CPLFW.txt", "flat", "cplfw"),
]

MODELS = ["arcface", "face_recognition"]


def run_all():
    lines = [f"Scorecard generated {datetime.now().isoformat(timespec='seconds')}", ""]

    for model in MODELS:
        lines.append(f"=== {model} ===")

        # First condition (LFW) runs with threshold=None so it finds its own EER
        # threshold; that value is then reused for every other condition.
        baseline_threshold = None

        for label, pairs_path, pairs_format, dataset_key in CONDITIONS:
            embeddings_path = EMBEDDINGS_DIR / f"{dataset_key}_{model}.json"
            result = score_condition(pairs_path, pairs_format, embeddings_path, baseline_threshold)

            if baseline_threshold is None:
                baseline_threshold = result["threshold"]

            lines.append(
                f"{label:20s} threshold={result['threshold']:.4f}  "
                f"accuracy={result['accuracy']:.4f}  "
                f"FAR={result['far']:.4f}  "
                f"FRR={result['frr']:.4f}  "
                f"pairs_scored={result['num_pairs_scored']}"
            )

        lines.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_all()

# Usage: python src/run_scorecard.py