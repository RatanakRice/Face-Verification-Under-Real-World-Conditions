"""
Degrades a folder of aligned face crops by blacking out a randomly sized,
randomly placed square box on one side of the face (seeded by filename).

Input: a folder of aligned crops, --size-min/--size-max (box size as a
fraction of width/height; both set to 0.40 for the reported results).
Output: one subfolder of masked images plus boxes.json (true box
coordinates per image, used later by landmark_mirror_fix.py).
Report section: Methodology / Data Processing and Perturbations.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def resolve_side(side: str, rng: random.Random) -> str:
    """
    Turn --side into "left"/"right". Always draws from rng, even when the side
    is fixed, so box sizes/positions are the same for every --side setting.
    """
    drawn = rng.choice(["left", "right"])
    return side if side in ("left", "right") else drawn


def place_box(width: int, height: int, side: str, size_range, rng: random.Random):
    """
    Randomly sized box at a random height, horizontally centred at 30% width
    (left) or 70% width (right). Never touches the image border (at least 1px
    away). Returns (x, y, w, h).
    """
    size_min, size_max = size_range
    w = max(1, round(rng.uniform(size_min, size_max) * width))
    h = max(1, round(rng.uniform(size_min, size_max) * height))

    cx = round(0.30 * width) if side == "left" else round(0.70 * width)
    x = max(1, min(cx - w // 2, width - w - 1))
    y = rng.randint(1, max(1, height - h - 1))
    return x, y, w, h


def folder_name(side: str, size_range) -> str:
    """block_{side}_{min}_{max}, e.g. block_15_30 ("random" drops the side segment)."""
    size_min, size_max = size_range
    tag = f"{round(size_min * 100)}_{round(size_max * 100)}"
    return f"block_{tag}" if side == "random" else f"block_{side}_{tag}"


def process_folder(image_paths, output_root: Path, side: str, size_range):
    """Mask every image, writing images plus boxes.json to output_root/<folder_name>/."""
    output_dir = output_root / folder_name(side, size_range)
    output_dir.mkdir(parents=True, exist_ok=True)

    boxes, skipped = {}, []
    for image_path in tqdm(image_paths, desc=f"Masking {output_dir.name}", unit="img"):
        image = cv2.imread(str(image_path))
        if image is None:
            skipped.append((image_path.name, "unreadable"))
            continue

        rng = random.Random(image_path.name)          # one generator per image, used for everything
        s = resolve_side(side, rng)
        height, width = image.shape[:2]
        x, y, w, h = place_box(width, height, s, size_range, rng)

        masked = image.copy()
        masked[y:y + h, x:x + w] = 0
        cv2.imwrite(str(output_dir / image_path.name), masked)
        boxes[image_path.name] = {"side": s, "box": [x, y, w, h]}

    with open(output_dir / "boxes.json", "w") as f:
        json.dump(boxes, f)

    print(f"Done. {len(skipped)} images skipped. Output: {output_dir}")
    for name, reason in skipped[:20]:
        print(f"  skipped: {name} ({reason})")
    if len(skipped) > 20:
        print(f"  ... and {len(skipped) - 20} more")


def main():
    parser = argparse.ArgumentParser(description="Mask a randomly sized box at a random height on one side of aligned face crops.")
    parser.add_argument("--input", required=True, help="Aligned crops folder, e.g. data/processed/aligned/lfw")
    parser.add_argument("--output-root", required=True, help="Root output folder, e.g. data/processed/occlusion/lfw")
    parser.add_argument("--side", choices=["left", "right", "random"], default="random",
                        help="Which side to mask. 'random' picks per image, seeded by filename. Default: random.")
    parser.add_argument("--size-min", type=float, default=0.15, help="Minimum box side, as a fraction of width/height. Default: 0.15.")
    parser.add_argument("--size-max", type=float, default=0.30, help="Maximum box side, as a fraction of width/height. Default: 0.30.")
    args = parser.parse_args()

    image_paths = sorted(Path(args.input).glob("*.jpg"))
    print(f"Found {len(image_paths)} aligned images in {args.input}")
    process_folder(image_paths, Path(args.output_root), args.side, (args.size_min, args.size_max))


if __name__ == "__main__":
    main()

# Usage: python src/occlusion.py --input data/processed/aligned/lfw --output-root data/processed/occlusion/lfw --size-min 0.15 --size-max 0.30