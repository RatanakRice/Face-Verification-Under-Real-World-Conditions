import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


# ArcFace 5-point reference landmarks: left eye, right eye, nose, left/right
# mouth corner. Defined for a 112x112 crop, scaled up to the size we save at.
OUTPUT_SIZE = (256, 256)
_SCALE = OUTPUT_SIZE[0] / 112.0
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
) * _SCALE


def load_detector(checkpoint_path: str) -> YOLO:
    """Load the face-tuned YOLOv8n weights."""
    return YOLO(checkpoint_path)


def detect_face(model: YOLO, image: np.ndarray):
    """
    Run detection on one image.

    Returns the 5 landmark points of the highest-confidence face,
    or None if no face (or no landmarks) is found. Alignment only
    needs the landmarks, so the bounding box is not returned.
    """
    results = model.predict(image, verbose=False)[0]

    if len(results.boxes) == 0 or getattr(results, "keypoints", None) is None:
        return None

    best_idx = int(results.boxes.conf.argmax())
    kpts = results.keypoints.xy[best_idx].cpu().numpy()
    if kpts.shape[0] < 5:
        return None

    return kpts[:5]


def align_face(image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Warp the detected 5 landmarks onto the ArcFace template, returning an OUTPUT_SIZE crop."""
    transform, _ = cv2.estimateAffinePartial2D(keypoints.astype(np.float32), ARCFACE_TEMPLATE)
    aligned = cv2.warpAffine(image, transform, OUTPUT_SIZE, borderValue=0)
    return aligned


def process_folder(model: YOLO, input_dir: Path, output_dir: Path):
    """Detect + align every .jpg under input_dir, writing flat crops to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(input_dir.rglob("*.jpg"))

    skipped = []
    for image_path in tqdm(image_paths, desc="Detecting + aligning", unit="img"):
        image = cv2.imread(str(image_path))
        if image is None:
            skipped.append((image_path, "unreadable"))
            continue

        keypoints = detect_face(model, image)
        if keypoints is None:
            skipped.append((image_path, "no face or landmarks detected"))
            continue

        aligned = align_face(image, keypoints)
        cv2.imwrite(str(output_dir / image_path.name), aligned)

    print(f"Done. {len(skipped)} images skipped.")
    for path, reason in skipped[:20]:
        print(f"  skipped: {path} ({reason})")
    if len(skipped) > 20:
        print(f"  ... and {len(skipped) - 20} more")


def main():
    parser = argparse.ArgumentParser(description="Detect + align faces for one dataset.")
    parser.add_argument("--checkpoint", required=True, help="Path to face-tuned YOLOv8n weights")
    parser.add_argument("--input", required=True, help="Raw images folder, e.g. data/raw/lfw")
    parser.add_argument("--output", required=True, help="Output folder, e.g. data/processed/aligned/lfw")
    args = parser.parse_args()

    model = load_detector(args.checkpoint)
    process_folder(model, Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()

# run: python src/detect_align.py --checkpoint model/yolov8n-face-derronqi.pt --input data/raw/lfw --output data/processed/aligned/lfw
