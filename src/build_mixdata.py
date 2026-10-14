"""
Builds one "mixed" dataset per source dataset: every image gets exactly one
randomly-assigned condition (clean / one of three lighting gammas / 40%
occlusion), rather than every image in a folder sharing the same condition.
Closer to a real, messy photo collection than the single-condition folders
used elsewhere in this project -- the point is to test quality_gate.py on a
pool where it can't assume what's wrong with any given photo.

The choice per image is seeded by filename (same convention as
occlusion.py's side pick), so re-running reproduces the identical mix.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm

CONDITIONS = ["clean", "gamma_0.4", "gamma_1.8", "gamma_2.5", "occlusion_40"]


def source_paths(processed_dir: Path, dataset: str) -> dict:
    """Map each condition name to the folder holding that condition's version
    of every image in `dataset`."""
    return {
        "clean": processed_dir / "aligned" / dataset,
        "gamma_0.4": processed_dir / "lighting" / dataset / "gamma_0.4",
        "gamma_1.8": processed_dir / "lighting" / dataset / "gamma_1.8",
        "gamma_2.5": processed_dir / "lighting" / dataset / "gamma_2.5",
        "occlusion_40": processed_dir / "occlusion" / dataset / "block_40_40",
    }


def build(processed_dir: Path, dataset: str, output_dir: Path, seed_prefix: str = ""):
    """For every image in the clean/aligned folder, copy in the version from
    one randomly (filename-seeded) chosen condition. Writes manifest.json
    recording which condition each image got."""
    sources = source_paths(processed_dir, dataset)
    image_names = sorted(p.name for p in sources["clean"].glob("*.jpg"))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    missing = []

    for name in tqdm(image_names, desc=f"Building mixdata_{dataset}", unit="img"):
        rng = random.Random(seed_prefix + name)
        condition = rng.choice(CONDITIONS)
        src = sources[condition] / name
        if not src.exists():
            missing.append((name, condition))
            continue
        shutil.copy(src, output_dir / name)
        manifest[name] = condition

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    counts = {c: sum(1 for v in manifest.values() if v == c) for c in CONDITIONS}
    print(f"Done. {len(manifest)} images written, {len(missing)} missing source files.")
    for name, condition in missing[:20]:
        print(f"  missing: {name} ({condition})")
    print(f"Condition counts: {counts}")


def main():
    parser = argparse.ArgumentParser(description="Build a per-image randomized mix of clean/lighting/occlusion conditions for one dataset.")
    parser.add_argument("--processed-dir", default="data2/processed", help="Root holding aligned/lighting/occlusion folders")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. lfw_bigger")
    parser.add_argument("--output", required=True, help="Output folder, e.g. data2/processed/mixdata_lfw_bigger")
    parser.add_argument("--seed-prefix", default="", help="Optional prefix mixed into the per-image seed, for building multiple distinct mixes")
    args = parser.parse_args()

    build(Path(args.processed_dir), args.dataset, Path(args.output), args.seed_prefix)


if __name__ == "__main__":
    main()

# Usage: python src/build_mixdata.py --dataset lfw_bigger --output data2/processed/mixdata_lfw_bigger
