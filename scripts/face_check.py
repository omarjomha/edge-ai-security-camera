"""Face quick-checks on still images (dev tool, not part of the live pipeline).

One image:  python -m scripts.face_check photo.jpg          -> boxes + aligned crops
Several:    python -m scripts.face_check a.jpg b.jpg c.jpg  -> similarity table
"""
import argparse
import os

import cv2
import numpy as np

from scripts.enroll import embed_largest_face
from src.face import ALIGNED_SIZE, FaceDetector, FaceEmbedder, align_face, draw_faces
from src.gallery import cosine_similarity


def show_one(face_det: FaceDetector, path: str, save: str, aligned_dir: str) -> None:
    """One image: print + draw every face, save aligned crops."""
    img = cv2.imread(path)
    boxes, landmarks = face_det.detect(img)

    print(f"image: {img.shape}  ->  {len(boxes)} faces")
    for i, ((x1, y1, x2, y2, score), pts) in enumerate(zip(boxes, landmarks)):
        print(f"  face {i}: score {score:.2f}  size {x2 - x1:.0f}x{y2 - y1:.0f}px  "
              f"left eye {pts[0].round()}  right eye {pts[1].round()}")

    cv2.imwrite(save, draw_faces(img, boxes, landmarks))
    print(f"saved: {save}")

    if len(landmarks):
        aligned = [align_face(img, pts) for pts in landmarks]
        os.makedirs(aligned_dir, exist_ok=True)
        for i, face in enumerate(aligned):
            cv2.imwrite(os.path.join(aligned_dir, f"face_{i}.jpg"), face)
        # All aligned faces side by side: eyes should line up across the whole strip
        strip_path = os.path.join(aligned_dir, "strip.jpg")
        cv2.imwrite(strip_path, np.hstack(aligned))
        print(f"saved {len(aligned)} aligned faces ({ALIGNED_SIZE}x{ALIGNED_SIZE}) + {strip_path}")


def compare(face_det: FaceDetector, embedder: FaceEmbedder, paths: list[str]) -> None:
    """Several images: embed the largest face in each, print a similarity table."""
    names, vecs = [], []
    for path in paths:
        vec = embed_largest_face(face_det, embedder, cv2.imread(path))
        if vec is None:
            print(f"  no face found in {path}, skipping")
            continue
        vecs.append(vec)
        names.append(os.path.splitext(os.path.basename(path))[0])

    width = max(len(n) for n in names) + 2
    print("cosine similarity (1.0 = identical, ~0 = unrelated):")
    print(" " * width + "".join(f"{n:>{width}}" for n in names))
    for name, a in zip(names, vecs):
        print(f"{name:<{width}}" + "".join(f"{cosine_similarity(a, b):>{width}.2f}" for b in vecs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--model", default="models/scrfd_500m.onnx")
    parser.add_argument("--embed-model", default="models/arcface_mbf.onnx")
    parser.add_argument("--save", default="results/faces.jpg")
    # Aligned crops are real faces -> default into data/ (gitignored), not results/
    parser.add_argument("--aligned-dir", default="data/aligned")
    args = parser.parse_args()

    face_det = FaceDetector(args.model)
    if len(args.images) == 1:
        show_one(face_det, args.images[0], args.save, args.aligned_dir)
    else:
        compare(face_det, FaceEmbedder(args.embed_model), args.images)
