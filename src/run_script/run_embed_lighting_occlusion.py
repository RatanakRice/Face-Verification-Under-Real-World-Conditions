"""
Embeds the lighting/occlusion condition folders under data/processed/ (the
earlier, smaller pilot runs -- see run_embed_bigger.py for the full-scale
"_bigger" equivalent).

Input: data/processed/lighting/, data/processed/occlusion/.
Output: data/processed/embeddings/.
Report section: pilot work behind the gamma/occlusion-size decisions in
Parameter Settings.
"""
from pathlib import Path

from embed import load_arcface_model, process_folder as embed_folder

LIGHTING_DIR = Path("data/processed/lighting")
OCCLUSION_DIR = Path("data/processed/occlusion")
EMBEDDINGS_DIR = Path("data/processed/embeddings")

DATASETS = ["lfw", "calfw", "cplfw", "lfw_bigger", "calfw_bigger", "cplfw_bigger"]

from constants import GAMMA_VALUES
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

# Usage: PYTHONPATH=src python3 src/run_script/run_embed_lighting_occlusion.py
