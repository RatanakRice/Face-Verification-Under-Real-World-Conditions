"""
Quality-aware preprocessing layer (the "+" between Align and Embed in the
pipeline diagram): one function that looks at an aligned face crop, decides
what's wrong with it, fixes it, or rejects it with a reason.

Order matters: occlusion is checked and fixed *first*, lighting *second*.
A solid black occlusion box drags the whole-image brightness average down,
which would bias the lighting estimate if lighting ran first; fixing (or
rejecting) the occlusion before estimating brightness means that estimate
always runs on a face with no degenerate black pixels left in it, and
requires no change to either estimate_gamma() or fix() -- just calling them
in this order.

Detection thresholds were calibrated empirically against a sample of clean,
undegraded photos, not guessed:
  - occlusion: a sample of clean crops found a largest-dark-box area of at
    most 2% of the image; the project's own 40% synthetic occlusion box is
    16%. OCCLUSION_AREA_THRESHOLD sits well clear of both.
  - lighting: the same clean sample's estimated corrective gamma ranged
    0.66-1.47 from ordinary photo-to-photo brightness variation alone (skin
    tone, exposure, background), overlapping the milder degraded conditions.
    GAMMA_OK_RANGE is set conservatively wide so it won't flag normal
    photos as degraded -- it will miss some mild lighting shifts, but mild
    shifts are exactly the ones shown elsewhere to barely hurt accuracy.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from auto_gamma_correct import auto_gamma_correct, estimate_gamma
from landmark_mirror_fix import find_box
from landmark_mirror_fix import fix as mirror_fix

OCCLUSION_AREA_THRESHOLD = 0.05
GAMMA_OK_RANGE = (0.70, 1.35)


def assess(image: np.ndarray, pairs: np.ndarray):
    """
    Inspect and fix one image. Returns (output_image, info).
    info always records what was found: "occlusion" / "lighting" (bool each),
    and "estimated_gamma" if lighting was assessed. If the image is rejected,
    output_image is None and info["rejected"] / info["reject_reason"] explain
    why -- currently only triggered when occlusion is detected but the
    mirror-fit can't be done with confidence (face not found, too few visible
    landmark pairs, or the affine fit failed), rather than silently handing
    back a lower-quality plain-inpaint patch.
    """
    info = {"occlusion": False, "lighting": False, "rejected": False}
    H, W = image.shape[:2]
    box = find_box(image)
    _, _, w, h = box
    has_occlusion = (w * h) / (H * W) > OCCLUSION_AREA_THRESHOLD

    working = image
    if has_occlusion:
        info["occlusion"] = True
        fixed, fix_info = mirror_fix(working, pairs, box=box, debug=True)
        if fix_info["method"] != "mirror":
            info["rejected"] = True
            info["reject_reason"] = (
                f"occlusion detected but not fixed confidently "
                f"({fix_info.get('reason', fix_info['method'])})"
            )
            return None, info
        working = fixed

    gamma = estimate_gamma(working)
    info["estimated_gamma"] = gamma
    if not (GAMMA_OK_RANGE[0] <= gamma <= GAMMA_OK_RANGE[1]):
        info["lighting"] = True
        working = auto_gamma_correct(working)

    return working, info


def process_folder(image_paths: list[Path], output_dir: Path, pairs: np.ndarray):
    """Run every image through assess(), writing fixed images to output_dir
    and a decisions.json log (per filename: occlusion/lighting flags,
    estimated gamma, reject reason if any)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    decisions = {}
    counts = {"occlusion_fixed": 0, "lighting_fixed": 0, "both": 0, "neither": 0, "rejected": 0}
    skipped = []

    for image_path in tqdm(image_paths, desc="Quality gate", unit="img"):
        image = cv2.imread(str(image_path))
        if image is None:
            skipped.append((image_path.name, "unreadable"))
            continue

        out, info = assess(image, pairs)
        decisions[image_path.name] = info

        if info["rejected"]:
            counts["rejected"] += 1
            continue

        cv2.imwrite(str(output_dir / image_path.name), out)
        if info["occlusion"] and info["lighting"]:
            counts["both"] += 1
        elif info["occlusion"]:
            counts["occlusion_fixed"] += 1
        elif info["lighting"]:
            counts["lighting_fixed"] += 1
        else:
            counts["neither"] += 1

    with open(output_dir / "decisions.json", "w") as f:
        json.dump(decisions, f)

    print(f"Done. {len(skipped)} images unreadable. Output: {output_dir}")
    for name, reason in skipped[:20]:
        print(f"  skipped: {name} ({reason})")
    print(f"Decisions: {counts}")


def main():
    parser = argparse.ArgumentParser(description="Quality-aware preprocessing gate: detect and fix (or reject) occlusion and bad lighting in one pass.")
    parser.add_argument("--input", required=True, help="Aligned crops folder")
    parser.add_argument("--output", required=True, help="Output folder (fixed images + decisions.json)")
    parser.add_argument("--pairs", default="model/mesh_pairs.npy", help="Landmark pairing table from landmark_mirror_fix.py --build-pairs")
    args = parser.parse_args()

    pairs = np.load(args.pairs)
    image_paths = sorted(Path(args.input).glob("*.jpg"))
    print(f"Found {len(image_paths)} images in {args.input}")
    process_folder(image_paths, Path(args.output), pairs)


if __name__ == "__main__":
    main()

# Usage: PYTHONPATH=src python3 src/quality_gate.py --input <aligned-or-degraded folder> --output <fixed folder>
