"""
Generates the lighting/occlusion degraded conditions under data/processed/
(the earlier, smaller pilot runs; mild occlusion sizes block_15_30/
block_30_30, superseded by the 40% box used in the final results).

Input: data/processed/aligned/.
Output: data/processed/lighting/, data/processed/occlusion/.
Report section: pilot work behind the occlusion-size retuning described in
Parameter Settings.
"""
from pathlib import Path

from lighting import process_folder as apply_lighting
from occlusion import process_folder as apply_occlusion

ALIGNED_DIR = Path("data/processed/aligned")
LIGHTING_DIR = Path("data/processed/lighting")
OCCLUSION_DIR = Path("data/processed/occlusion")

DATASETS = ["calfw", "cplfw", "lfw_bigger", "calfw_bigger", "cplfw_bigger"]

from constants import GAMMA_VALUES
OCCLUSION_CONFIGS = [("random", 0.25), ("random", 0.5)]


def run_all():
    for dataset in DATASETS:
        input_dir = ALIGNED_DIR / dataset
        image_paths = sorted(input_dir.glob("*.jpg"))
        print(f"\n=== {dataset}: {len(image_paths)} aligned images ===")

        lighting_root = LIGHTING_DIR / dataset
        for gamma in GAMMA_VALUES:
            print(f"\n--- {dataset}: lighting gamma={gamma} ---")
            apply_lighting(image_paths, lighting_root, gamma)

        occlusion_root = OCCLUSION_DIR / dataset
        for side, mask_fraction in OCCLUSION_CONFIGS:
            print(f"\n--- {dataset}: occlusion side={side} mask_fraction={mask_fraction} ---")
            apply_occlusion(image_paths, occlusion_root, side, mask_fraction)

    print("\nAll datasets done.")


if __name__ == "__main__":
    run_all()

# Usage: PYTHONPATH=src python3 src/run_script/run_lighting_occlusion.py
