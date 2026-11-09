"""Run the live pipeline on still images and save annotated copies (e.g. for the report).

Usage: python -m scripts.annotate_images data/test_faces --out results/test_faces
Uses the exact per-frame pipeline of src/main.py. A still image can't be tracked from
one frame, so each image is fed as if held in front of a 15 fps camera until just past
the grace period (fresh tracker + state machine per image); the figure shows the verdict.
"""
import argparse
import os

import cv2
import yaml

from src.detector import Detector
from src.face import FaceDetector, FaceEmbedder
from src.gallery import load_gallery
from src.main import draw_frame, process_frame
from src.state_machine import StateMachine
from src.tracker import Tracker

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
        tracker = Tracker(cfg["track_high_conf"], cfg["track_new_conf"], cfg["track_match_iou"],
                          cfg["track_max_misses"], cfg["track_min_hits"], cfg["vote_window"])
        sm = StateMachine(cfg["grace_period_s"], cfg["vote_min_matches"])
        # "Hold still" just past the grace period at 15 fps, so every person has a verdict
        n_frames = int((cfg["grace_period_s"] + 1) * 15)
        for i in range(n_frames):
            out = process_frame(img, detector, face_det, embedder, gallery, tracker, sm, i / 15, cfg)
        vis = draw_frame(img, out, sm, (n_frames - 1) / 15)

        out_path = os.path.join(args.out, os.path.splitext(fname)[0] + "_annotated.jpg")
        cv2.imwrite(out_path, vis)
        states = {tid: p.state + (f" {p.friend}" if p.friend else "") for tid, p in sm.people.items()}
        print(f"{fname:<14} -> {states}  saved {out_path}")


if __name__ == "__main__":
    main()
