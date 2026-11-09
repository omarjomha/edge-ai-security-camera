"""FP16 accuracy check: do the face models give the same answers on CPU and on the Orin GPU?

Usage (on the Orin): python -m scripts.check_backend_parity data/test_faces/*.jpg
Runs every photo through SCRFD + ArcFace twice, once on the CPU (FP32 reference) and once
on the backend in config.yaml (e.g. tensorrt fp16), then compares:
  1. each photo's two embeddings (cosine similarity, should be ~0.99+)
  2. the person-vs-person similarity table (differences should be ~0.01), so the
     similarity_threshold tuned on the laptop still holds
"""
import argparse
import os

import cv2
import numpy as np
import yaml

from src.face import FaceDetector, FaceEmbedder, align_face
from src.gallery import cosine_similarity
from src.runtime import backend_label, configure


def embed_all(cfg: dict, paths: list[str]) -> dict[str, np.ndarray]:
    """Largest face of each photo -> embedding, on whatever backend is configured now."""
    face_det = FaceDetector(cfg["face_det_model"], conf=cfg["face_conf"])
    embedder = FaceEmbedder(cfg["face_embed_model"])
    vecs = {}
    for path in paths:
        img = cv2.imread(path)
        boxes, landmarks = face_det.detect(img)
        if len(boxes) == 0:
            print(f"  no face in {path}, skipping")
            continue
        biggest = int(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).argmax())
        vecs[os.path.splitext(os.path.basename(path))[0]] = embedder.embed(
            align_face(img, landmarks[biggest]))
    return vecs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    configure({**cfg, "backend": "cpu"})
    ref = embed_all(cfg, args.images)
    configure(cfg)
    test = embed_all(cfg, args.images)
    label = backend_label()
    names = [n for n in ref if n in test]

    print(f"1. same photo, cpu vs {label} (1.00 = identical):")
    for n in names:
        print(f"   {n:<16} {cosine_similarity(ref[n], test[n]):.4f}")

    print(f"2. similarity table, {label} (cpu value in brackets):")
    worst = 0.0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            s_ref, s_test = cosine_similarity(ref[a], ref[b]), cosine_similarity(test[a], test[b])
            worst = max(worst, abs(s_test - s_ref))
            print(f"   {a} vs {b}: {s_test:.3f} ({s_ref:.3f})")
    print(f"   largest difference: {worst:.3f} (threshold is {cfg['similarity_threshold']})")


if __name__ == "__main__":
    main()
