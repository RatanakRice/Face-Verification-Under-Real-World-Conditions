import argparse
import os
from pathlib import Path

from embed import load_arcface_model, process_folder as embed_folder


def find_bigger_image_folders(root: Path):
    """
    Yield every directory under root that directly contains .jpg files and has
    a "_bigger" dataset folder somewhere in its path, e.g.:
      aligned/calfw_bigger
      lighting/calfw_bigger/gamma_0.4
      occlusion/calfw_bigger/block_15_30
    """
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)

        if not any(name.endswith(".jpg") for name in filenames):
            continue

        rel_parts = current.relative_to(root).parts
        if any(part.endswith("_bigger") for part in rel_parts):
            yield current


def already_embedded(output_dir: Path) -> bool:
    return (output_dir / "arcface.json").exists() and (output_dir / "face_recognition.json").exists()


def run_all(processed_dir: Path, embeddings_dir: Path, skip_existing: bool = True):
    arcface_model = load_arcface_model()

    folders = sorted(find_bigger_image_folders(processed_dir))
    print(f"Found {len(folders)} '_bigger' image folders to embed.\n")

    for input_dir in folders:
        rel = input_dir.relative_to(processed_dir)
        output_dir = embeddings_dir / rel

        if skip_existing and already_embedded(output_dir):
            print(f"=== {rel} === skipped (already embedded)")
            continue

        print(f"\n=== {rel} ===")
        embed_folder(arcface_model, input_dir, output_dir, str(rel), file_prefix="")

    print("\nAll '_bigger' folders embedded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed every '_bigger' image folder under a data root.")
    parser.add_argument("--processed-dir", default="data/processed", help="Root to scan for '_bigger' image folders")
    parser.add_argument("--embeddings-dir", default="data/embeddings", help="Root to mirror embeddings into")
    args = parser.parse_args()
    run_all(Path(args.processed_dir), Path(args.embeddings_dir))

# Usage: PYTHONPATH=src python3 src/run_script/run_embed_bigger.py [--processed-dir data2/processed --embeddings-dir data2/embeddings]
# Output layout mirrors the input: <embeddings-dir>/<same relative path>/{arcface,face_recognition}.json
