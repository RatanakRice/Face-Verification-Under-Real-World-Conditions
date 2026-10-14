import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from lighting import apply_gamma

TARGET_MEAN = 128.0
GAMMA_MIN, GAMMA_MAX = 0.05, 10.0


def estimate_gamma(image: np.ndarray, target_mean: float = TARGET_MEAN) -> float:
    """
    Estimate the gamma that would bring image's mean brightness to target_mean,
    without knowing how the image was originally darkened/brightened.

    Solves target_mean/255 = (mean/255)**gamma for gamma. Clamped so a near-black
    or near-white image (mean close to 0 or 255) can't blow up to an extreme value.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = max(float(gray.mean()), 1e-3)
    gamma = np.log(target_mean / 255.0) / np.log(mean / 255.0)
    return float(np.clip(gamma, GAMMA_MIN, GAMMA_MAX))


def auto_gamma_correct(image: np.ndarray, target_mean: float = TARGET_MEAN) -> np.ndarray:
    """Estimate and apply the corrective gamma in one step."""
    gamma = estimate_gamma(image, target_mean)
    return apply_gamma(image, gamma)


def process_folder(image_paths: list[Path], output_dir: Path, target_mean: float = TARGET_MEAN):
    """Auto-correct every image's brightness, writing to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped = []
    for image_path in tqdm(image_paths, desc="Auto-gamma correcting", unit="img"):
        image = cv2.imread(str(image_path))
        if image is None:
            skipped.append((image_path.name, "unreadable"))
            continue

        corrected = auto_gamma_correct(image, target_mean)
        cv2.imwrite(str(output_dir / image_path.name), corrected)

    print(f"Done. {len(skipped)} images skipped. Output: {output_dir}")
    for name, reason in skipped[:20]:
        print(f"  skipped: {name} ({reason})")
    if len(skipped) > 20:
        print(f"  ... and {len(skipped) - 20} more")


def main():
    parser = argparse.ArgumentParser(description="Auto-correct gamma/brightness on lighting-degraded face crops.")
    parser.add_argument("--input", required=True, help="Degraded crops folder, e.g. data/processed/lighting/lfw/gamma_2.5")
    parser.add_argument("--output", required=True, help="Output folder, e.g. data/processed/lighting_corrected/lfw/gamma_2.5")
    parser.add_argument("--target-mean", type=float, default=TARGET_MEAN, help=f"Target grayscale mean brightness. Default: {TARGET_MEAN}.")
    args = parser.parse_args()

    image_paths = sorted(Path(args.input).glob("*.jpg"))
    print(f"Found {len(image_paths)} images in {args.input}")
    process_folder(image_paths, Path(args.output), args.target_mean)


if __name__ == "__main__":
    main()

# Usage: python src/auto_gamma_correct.py --input data/processed/lighting/lfw/gamma_2.5 --output data/processed/lighting_corrected/lfw/gamma_2.5
