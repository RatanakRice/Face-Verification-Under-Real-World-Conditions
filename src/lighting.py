import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply a gamma lighting shift on pixels normalised to [0, 1].

    gamma > 1 darkens (mid-tones pushed down), gamma < 1 brightens.
    """
    normalized = image.astype(np.float32) / 255.0
    shifted = np.power(normalized, gamma)
    return (shifted * 255.0).clip(0, 255).astype(np.uint8)


def process_folder(image_paths: list[Path], output_root: Path, gamma: float):
    """Apply one gamma value to every image, writing to output_root/gamma_{gamma}/."""
    output_dir = output_root / f"gamma_{gamma}"
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped = []
    for image_path in tqdm(image_paths, desc=f"Applying gamma={gamma}", unit="img"):
        image = cv2.imread(str(image_path))
        if image is None:
            skipped.append((image_path.name, "unreadable"))
            continue

        shifted = apply_gamma(image, gamma)
        cv2.imwrite(str(output_dir / image_path.name), shifted)

    print(f"Done. {len(skipped)} images skipped. Output: {output_dir}")
    for name, reason in skipped[:20]:
        print(f"  skipped: {name} ({reason})")
    if len(skipped) > 20:
        print(f"  ... and {len(skipped) - 20} more")


def main():
    parser = argparse.ArgumentParser(description="Apply a gamma lighting shift to aligned face crops, one output subfolder per gamma value.")
    parser.add_argument("--input", required=True, help="Aligned crops folder, e.g. data/processed/aligned/lfw")
    parser.add_argument("--output-root", required=True, help="Root output folder, e.g. data/processed/lighting/lfw")
    parser.add_argument("--gamma-values", type=float, nargs="+", default=[1.8], help="One or more gamma values. >1 darkens, <1 brightens. Default: [1.8].")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_root = Path(args.output_root)

    image_paths = sorted(input_dir.glob("*.jpg"))
    print(f"Found {len(image_paths)} aligned images in {input_dir}")

    for gamma in args.gamma_values:
        print(f"\n--- gamma={gamma} ---")
        process_folder(image_paths, output_root, gamma)


if __name__ == "__main__":
    main()

# run: python src/lighting.py --input data/processed/aligned/lfw --output-root data/processed/lighting/lfw --gamma-values 1.8
