import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def resolve_side(side: str, image_name: str) -> str:
    """
    Turn the --side option into a concrete "left"/"right" for one image.

    "random" picks per image, seeded by the filename so a re-run masks the
    same side for the same image.
    """
    if side in ("left", "right"):
        return side
    return random.Random(image_name).choice(["left", "right"])


def apply_mask(image: np.ndarray, side: str, mask_fraction: float) -> np.ndarray:
    """
    Black out a vertical strip on one side of the image.

    "left" masks the leftmost mask_fraction of the width, "right" masks the rightmost.
    """
    masked = image.copy()
    height, width = image.shape[:2]
    strip_width = int(width * mask_fraction)

    if side == "left":
        masked[:, :strip_width] = 0
    else:
        masked[:, width - strip_width:] = 0

    return masked


def process_folder(image_paths: list[Path], output_root: Path, side: str, mask_fraction: float):
    """Apply one mask config to every image, writing to output_root/{side}_{mask_fraction}/."""
    output_dir = output_root / f"{side}_{mask_fraction}"
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped = []
    for image_path in tqdm(image_paths, desc=f"Masking {side} {mask_fraction}", unit="img"):
        image = cv2.imread(str(image_path))
        if image is None:
            skipped.append((image_path.name, "unreadable"))
            continue

        masked = apply_mask(image, resolve_side(side, image_path.name), mask_fraction)
        cv2.imwrite(str(output_dir / image_path.name), masked)

    print(f"Done. {len(skipped)} images skipped. Output: {output_dir}")
    for name, reason in skipped[:20]:
        print(f"  skipped: {name} ({reason})")
    if len(skipped) > 20:
        print(f"  ... and {len(skipped) - 20} more")


def main():
    parser = argparse.ArgumentParser(description="Mask a vertical strip on one side of aligned face crops.")
    parser.add_argument("--input", required=True, help="Aligned crops folder, e.g. data/processed/aligned/lfw")
    parser.add_argument("--output-root", required=True, help="Root output folder, e.g. data/processed/occlusion/lfw")
    parser.add_argument("--side", choices=["left", "right", "random"], default="random",
                        help="Which side to mask. 'random' picks per image, seeded by filename. Default: random.")
    parser.add_argument("--mask-fraction", type=float, default=0.25, help="Fraction of image width to mask. Default: 0.25.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_root = Path(args.output_root)

    image_paths = sorted(input_dir.glob("*.jpg"))
    print(f"Found {len(image_paths)} aligned images in {input_dir}")

    process_folder(image_paths, output_root, args.side, args.mask_fraction)


if __name__ == "__main__":
    main()

# Usage: python src/occlusion.py --input data/processed/aligned/lfw --output-root data/processed/occlusion/lfw --side random --mask-fraction 0.25
