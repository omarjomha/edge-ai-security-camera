"""Enroll friends: data/enroll/<name>/ photos -> gallery/gallery.npz.

Usage: python -m scripts.enroll
"""
import argparse
import os

import cv2
import numpy as np
import yaml

from src.face import FaceDetector, FaceEmbedder, align_face
from src.gallery import cosine_similarity, make_template, save_gallery

ENROLL_DIR = "data/enroll"   # gitignored: one subfolder of photos per friend
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
OUTLIER_SCORE = 0.5          # photo this far from its own template is probably bad


def embed_largest_face(face_det: FaceDetector, embedder: FaceEmbedder,
                       img: np.ndarray) -> np.ndarray | None:
    """Photo -> embedding of its biggest face (ignores people in the background).
    Returns None if no face is found. Enrollment photos = one main person each."""
    boxes, landmarks = face_det.detect(img)
    if len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return embedder.embed(align_face(img, landmarks[int(areas.argmax())]))


def embed_folder(face_det: FaceDetector, embedder: FaceEmbedder,
                 folder: str) -> tuple[list[str], list[np.ndarray]]:
    """Every photo in one friend's folder -> (filenames, embeddings)."""
    names, vecs = [], []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        vec = embed_largest_face(face_det, embedder, cv2.imread(os.path.join(folder, fname)))
        if vec is None:
            print(f"  no face found in {fname}, skipping")
            continue
        names.append(fname)
        vecs.append(vec)
    return names, vecs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    face_det = FaceDetector(cfg["face_det_model"], conf=cfg["face_conf"])  # 640 px: full photos
    embedder = FaceEmbedder(cfg["face_embed_model"])

    gallery = {}
    for name in sorted(os.listdir(ENROLL_DIR)):          # folder name = friend's name
        folder = os.path.join(ENROLL_DIR, name)
        if not os.path.isdir(folder):
            continue
        fnames, vecs = embed_folder(face_det, embedder, folder)
        if not vecs:
            print(f"{name}: no usable faces, not enrolled")
            continue
        gallery[name] = make_template(vecs)

        # Quality report: how well each photo agrees with the template
        print(f"{name}: {len(vecs)} photos")
        for fname, vec in zip(fnames, vecs):
            score = cosine_similarity(vec, gallery[name])
            flag = "  <-- outlier? check this photo" if score < OUTLIER_SCORE else ""
            print(f"  {fname:<20} similarity to template {score:.2f}{flag}")

    save_gallery(cfg["gallery_path"], gallery)
    print(f"saved {len(gallery)} friend(s) to {cfg['gallery_path']}")


if __name__ == "__main__":
    main()
