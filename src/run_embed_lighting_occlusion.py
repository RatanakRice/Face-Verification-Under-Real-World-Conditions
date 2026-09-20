"""
Embed every lighting- and occlusion-altered folder for all 6 datasets
(lfw, calfw, cplfw, lfw_bigger, calfw_bigger, cplfw_bigger):
3 gamma values + 2 occlusion severities each = 18 lighting + 12 occlusion
= 30 folders total.

Loads the ArcFace model once and reuses it across all 30 folders. Runs
sequentially in one process (CPU-bound, dlib-bound at ~20 img/s, no benefit
to parallelizing). Output mirrors the baseline embeddings layout but under
lighting/ and occlusion/ subdirectories, so baseline files
(data/processed/embeddings/lfw_arcface.json etc.) are untouched:

    data/processed/embeddings/lighting/{dataset}_gamma_{gamma}_{model}.json
    data/processed/embeddings/occlusion/{dataset}_{side}_{mask_fraction}_{model}.json

Usage:
    python src/run_embed_lighting_occlusion.py
"""

from pathlib import Path

from embed import load_arcface_model, process_folder as embed_folder

LIGHTING_DIR = Path("data/processed/lighting")
OCCLUSION_DIR = Path("data/processed/occlusion")
EMBEDDINGS_DIR = Path("data/processed/embeddings")

DATASETS = ["lfw", "calfw", "cplfw", "lfw_bigger", "calfw_bigger", "cplfw_bigger"]

GAMMA_VALUES = [0.4, 1.8, 2.5]
OCCLUSION_CONFIGS = [("random", 0.25), ("random", 0.5)]


def run_all():
    arcface_model = load_arcface_model()

    for dataset in DATASETS:
        for gamma in GAMMA_VALUES:
            input_dir = LIGHTING_DIR / dataset / f"gamma_{gamma}"
            name = f"{dataset}_gamma_{gamma}"
            print(f"\n=== lighting: {name} ===")
            embed_folder(arcface_model, input_dir, EMBEDDINGS_DIR / "lighting", name)

        for side, mask_fraction in OCCLUSION_CONFIGS:
            input_dir = OCCLUSION_DIR / dataset / f"{side}_{mask_fraction}"
            name = f"{dataset}_{side}_{mask_fraction}"
            print(f"\n=== occlusion: {name} ===")
            embed_folder(arcface_model, input_dir, EMBEDDINGS_DIR / "occlusion", name)

    print("\nAll 30 folders embedded.")


if __name__ == "__main__":
    run_all()

# run: python src/run_embed_lighting_occlusion.py
