"""Run the live pipeline on still images and save annotated copies (e.g. for the report).

Usage: python -m scripts.annotate_images data/test_faces --out results/test_faces
Uses the exact functions src/main.py runs per frame, so the labels match the live demo.
"""
import argparse
import os

import cv2
import yaml

from src.detector import Detector, draw
from src.face import FaceDetector, FaceEmbedder, detect_in_persons
from src.gallery import load_gallery
from src.main import draw_identities, identify_faces

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
MAX_SIDE = 1280              # output long side in px (close to the live 720p camera)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    detector = Detector(cfg["detector_model"], conf=cfg["detector_conf"])
    face_det = FaceDetector(cfg["face_det_model"], conf=cfg["face_conf"],
                            input_size=cfg["face_det_size"])
    embedder = FaceEmbedder(cfg["face_embed_model"])
    gallery = load_gallery(cfg["gallery_path"])
    os.makedirs(args.out, exist_ok=True)

    for fname in sorted(os.listdir(args.folder)):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        img = cv2.imread(os.path.join(args.folder, fname))
        # Phone photos can be 4000 px; shrink to camera-like size so labels are readable
        scale = MAX_SIDE / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        dets = detector.detect(img)
        boxes, landmarks = detect_in_persons(face_det, img, dets)
        identities = identify_faces(embedder, img, boxes, landmarks, gallery,
                                    cfg["similarity_threshold"], cfg["min_face_px"])
        vis = draw_identities(draw(img, dets), boxes, identities)

        out_path = os.path.join(args.out, os.path.splitext(fname)[0] + "_annotated.jpg")
        cv2.imwrite(out_path, vis)
        print(f"{fname:<14} -> {[label for label, _ in identities]}  saved {out_path}")


if __name__ == "__main__":
    main()
