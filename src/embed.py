import argparse
import json
import os
from pathlib import Path

import cv2
import face_recognition
import numpy as np
from insightface.model_zoo import get_model
from insightface.utils import ensure_available
from tqdm import tqdm

ARCFACE_PACK = "buffalo_l"
ARCFACE_RECOGNITION_ONNX = "w600k_r50.onnx"


def load_arcface_model():
    """
    Load the ArcFace recognition model from the buffalo_l pack.

    get_model() can't take a bare pack name (the pack holds several .onnx files),
    so we download the pack if needed and point it at the recognition net.
    """
    model_dir = ensure_available("models", ARCFACE_PACK, root="~/.insightface")
    model_path = os.path.join(model_dir, ARCFACE_RECOGNITION_ONNX)
    model = get_model(model_path)
    model.prepare(ctx_id=-1)  # CPU; set to 0 for a CUDA GPU
    return model


def embed_arcface(model, image: np.ndarray) -> np.ndarray:
    """Embed one aligned image with ArcFace. buffalo_l expects 112x112 input."""
    resized = cv2.resize(image, (112, 112))
    embedding = model.get_feat(resized)
    return embedding.flatten()


def embed_face_recognition(image: np.ndarray):
    """
    Embed one aligned image with face_recognition (dlib).

    The crop is already a face, so we pass the whole image as the face box.
    Returns None if dlib finds no landmarks on the crop.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb_image.shape[:2]
    whole_image_box = [(0, width, height, 0)]  # (top, right, bottom, left)

    encodings = face_recognition.face_encodings(rgb_image, known_face_locations=whole_image_box)
    if len(encodings) == 0:
        return None
    return encodings[0]


def process_folder(arcface_model, input_dir: Path, output_dir: Path, dataset_name: str):
    """Embed every image in input_dir with both models, saving one JSON file per model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(input_dir.glob("*.jpg"))

    arcface_embeddings = {}
    face_rec_embeddings = {}
    skipped = []

    for image_path in tqdm(image_paths, desc=f"Embedding {dataset_name}", unit="img"):
        image = cv2.imread(str(image_path))
        if image is None:
            skipped.append((image_path.name, "unreadable"))
            continue

        arcface_vector = embed_arcface(arcface_model, image)
        arcface_embeddings[image_path.name] = arcface_vector.tolist()

        face_rec_vector = embed_face_recognition(image)
        if face_rec_vector is None:
            skipped.append((image_path.name, "face_recognition found no landmarks"))
        else:
            face_rec_embeddings[image_path.name] = face_rec_vector.tolist()

    arcface_path = output_dir / f"{dataset_name}_arcface.json"
    face_rec_path = output_dir / f"{dataset_name}_face_recognition.json"

    with open(arcface_path, "w") as f:
        json.dump(arcface_embeddings, f)
    with open(face_rec_path, "w") as f:
        json.dump(face_rec_embeddings, f)

    print(f"Done. {len(skipped)} images skipped.")
    for name, reason in skipped[:20]:
        print(f"  skipped: {name} ({reason})")
    if len(skipped) > 20:
        print(f"  ... and {len(skipped) - 20} more")
    print(f"ArcFace embeddings: {len(arcface_embeddings)} -> {arcface_path}")
    print(f"face_recognition embeddings: {len(face_rec_embeddings)} -> {face_rec_path}")


def main():
    parser = argparse.ArgumentParser(description="Embed aligned face crops with ArcFace and face_recognition.")
    parser.add_argument("--input", required=True, help="Aligned crops folder, e.g. data/processed/aligned/lfw")
    parser.add_argument("--output", required=True, help="Where to save embedding files, e.g. data/processed/embeddings")
    parser.add_argument("--name", required=True, help="Dataset name used in output filenames, e.g. lfw")
    args = parser.parse_args()

    arcface_model = load_arcface_model()
    process_folder(arcface_model, Path(args.input), Path(args.output), args.name)


if __name__ == "__main__":
    main()

# run: python src/embed.py --input data/processed/aligned/lfw --output data/processed/embeddings --name lfw
