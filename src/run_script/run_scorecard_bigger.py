"""
Scores the full-scale ("_bigger") clean baseline and the milder,
superseded occlusion pilot sizes (block_15_30/block_30_30) from
data2/embeddings/.

Input: data2/embeddings/ (built by run_embed_bigger.py).
Output: data2/processed/scorecard_bigger.txt.
Report section: Table I clean baseline. (The reported 40%-box occlusion
and lighting-correction rows come from separate scripts -- see README.)
"""
import argparse
from datetime import datetime
from pathlib import Path

from compare import score_condition

PAIRS_DIR = Path("data/pairs")

DATASETS = ["lfw_bigger", "calfw_bigger", "cplfw_bigger"]
from constants import GAMMA_VALUES
OCCLUSION_CONFIGS = ["block_15_30", "block_30_30"]
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


def fmt_line(label: str, result: dict) -> str:
    return (
        f"{label:35s} accuracy={result['accuracy']:.4f}  "
        f"FAR={result['far']:.4f}  FRR={result['frr']:.4f}  "
        f"pairs_scored={result['num_pairs_scored']}"
    )


def run_all(embeddings_dir: Path, output_path: Path):
    lines = [f"'_bigger' embeddings scorecard generated {datetime.now().isoformat(timespec='seconds')}", ""]

    for model in MODELS:
        lines.append(f"=== {model} ===")

        # Threshold comes from the clean LFW_bigger baseline's own EER, then
        # reused for every other condition so results are comparable.
        baseline_pairs_path, baseline_pairs_format = PAIRS_BY_FAMILY["lfw"]
        baseline_embeddings_path = embeddings_dir / "aligned" / "lfw_bigger" / f"{model}.json"
        baseline_result = score_condition(baseline_pairs_path, baseline_pairs_format, baseline_embeddings_path, threshold=None)
        threshold = baseline_result["threshold"]
        lines.append(f"(threshold from LFW_bigger baseline EER: {threshold:.4f})")

        for dataset in DATASETS:
            pairs_path, pairs_format = PAIRS_BY_FAMILY[family_of(dataset)]

            label = f"{dataset} baseline (aligned)"
            embeddings_path = embeddings_dir / "aligned" / dataset / f"{model}.json"
            result = score_condition(pairs_path, pairs_format, embeddings_path, threshold)
            lines.append(fmt_line(label, result))

            for gamma in GAMMA_VALUES:
                label = f"{dataset} lighting gamma={gamma}"
                embeddings_path = embeddings_dir / "lighting" / dataset / f"gamma_{gamma}" / f"{model}.json"
                result = score_condition(pairs_path, pairs_format, embeddings_path, threshold)
                lines.append(fmt_line(label, result))

            for block in OCCLUSION_CONFIGS:
                label = f"{dataset} occlusion {block}"
                embeddings_path = embeddings_dir / "occlusion" / dataset / block / f"{model}.json"
                result = score_condition(pairs_path, pairs_format, embeddings_path, threshold)
                lines.append(fmt_line(label, result))

        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score every '_bigger' embedding folder against a fixed LFW-EER threshold.")
    parser.add_argument("--embeddings-dir", default="data/embeddings", help="Root holding aligned/lighting/occlusion embeddings")
    parser.add_argument("--output", default="data/processed/scorecard_bigger.txt", help="Where to write the scorecard text file")
    args = parser.parse_args()
    run_all(Path(args.embeddings_dir), Path(args.output))

# Usage: PYTHONPATH=src python3 src/run_script/run_scorecard_bigger.py [--embeddings-dir data2/embeddings --output data2/processed/scorecard_bigger.txt]
