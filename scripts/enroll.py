"""Dev enrollment: data/enroll/<name>/ photos -> gallery/gallery.npz.

Usage: python -m scripts.enroll
The enrollment logic is in src/enrollment.py; this script only reads folders,
prints the report and saves. Rebuilds the whole gallery from the photos each run.
"""
import argparse
import os

import cv2
import yaml

from src.enrollment import enroll_person
from src.face import FaceDetector, FaceEmbedder
from src.gallery import save_gallery

ENROLL_DIR = "data/enroll"   # gitignored: one subfolder of photos per friend
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


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
        photos = [(f, cv2.imread(os.path.join(folder, f)))
                  for f in sorted(os.listdir(folder)) if f.lower().endswith(IMAGE_EXTS)]
        template, results = enroll_person(face_det, embedder, photos,
                                          min_face_px=cfg["min_face_px"],
                                          min_photos=cfg["enroll_min_photos"],
                                          outlier_score=cfg["enroll_outlier_score"])

        n_ok = sum(r.accepted for r in results)
        status = "ENROLLED" if template is not None else "NOT ENROLLED"
        print(f"{name}: {status} ({n_ok}/{len(results)} photos used, "
              f"need {cfg['enroll_min_photos']})")
        for r in results:
            score = f"{r.score:.2f}" if r.score is not None else "  - "
            print(f"  {r.photo_id:<20} {score}  {'ok' if r.accepted else 'REJECTED: ' + r.reason}")
        if template is not None:
            gallery[name] = template

    save_gallery(cfg["gallery_path"], gallery)
    print(f"saved {len(gallery)} friend(s) to {cfg['gallery_path']}")


if __name__ == "__main__":
    main()
