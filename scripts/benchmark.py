"""Benchmark the full pipeline on a video file: per-stage latency + fps, saved as JSON.

Usage: python -m scripts.benchmark data/test_clips/clip.mp4 [--warmup 20]
Runs src/main.py's process_frame on every frame (no window). The first --warmup frames
are excluded (first runs are slow: memory allocation, GPU clocks ramping up).
Results are labelled with the device and backend, e.g. results/benchmark_<device>_<backend>.json
"""
import argparse
import json
import os
import platform
import time

import cv2
import numpy as np
import yaml

from src.detector import Detector
from src.face import FaceDetector, FaceEmbedder
from src.gallery import load_gallery
from src.main import process_frame
from src.runtime import backend_label, configure, describe, device_name
from src.state_machine import StateMachine
from src.tracker import Tracker


def stats(values: list[float]) -> dict:
    a = np.array(values)
    return {"mean": round(float(a.mean()), 2), "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2), "max": round(float(a.max()), 2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--warmup", type=int, default=20, help="frames excluded from the stats")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    configure(cfg)
    t0 = time.perf_counter()
    detector = Detector(cfg["detector_model"], conf=cfg["detector_conf"])
    face_det = FaceDetector(cfg["face_det_model"], conf=cfg["face_conf"],
                            input_size=cfg["face_det_size"])
    embedder = FaceEmbedder(cfg["face_embed_model"])
    load_s = time.perf_counter() - t0          # includes TensorRT engine build on a first run
    gallery = load_gallery(cfg["gallery_path"])
    tracker = Tracker(cfg["track_high_conf"], cfg["track_new_conf"], cfg["track_match_iou"],
                      cfg["track_max_misses"], cfg["track_min_hits"], cfg["vote_window"])
    sm = StateMachine(cfg["grace_period_s"], cfg["vote_min_matches"])
    print(f"{device_name()} | backend {backend_label()} | YOLO {describe(detector.session)} | "
          f"SCRFD {describe(face_det.session)} | ArcFace {describe(embedder.session)} | "
          f"model load {load_s:.1f} s")

    cap = cv2.VideoCapture(args.video)
    stages = {"detect": [], "face": [], "recog": [], "other": [], "total": []}
    people, n, frame_size = [], 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        frame_size = list(frame.shape[:2])     # (height, width)
        t = time.perf_counter()
        out = process_frame(frame, detector, face_det, embedder, gallery, tracker, sm, now, cfg)
        total = (time.perf_counter() - t) * 1000
        n += 1
        if n <= args.warmup:
            continue
        times = out["times"]
        for k in ("detect", "face", "recog"):
            stages[k].append(times[k])
        # tracker, face-to-track assignment, votes, state machine (all CPU / Python)
        stages["other"].append(total - sum(times.values()))
        stages["total"].append(total)
        people.append(len(out["tracks"]))
    cap.release()
    if not stages["total"]:
        raise SystemExit(f"Only {n} frames: need more than --warmup ({args.warmup})")

    result = {
        "device": device_name(), "backend": backend_label(), "machine": platform.machine(),
        "video": os.path.basename(args.video), "frame_size": frame_size,
        "frames_measured": len(stages["total"]), "warmup_frames": args.warmup,
        "avg_people_per_frame": round(float(np.mean(people)), 2),
        "model_load_s": round(load_s, 1),
        "latency_ms": {k: stats(v) for k, v in stages.items()},
        "fps": round(1000 / float(np.mean(stages["total"])), 1),
    }
    print(f"{result['frames_measured']} frames, avg {result['avg_people_per_frame']} people/frame")
    for k, s in result["latency_ms"].items():
        print(f"  {k:<7} mean {s['mean']:6.1f} ms | p50 {s['p50']:6.1f} | p95 {s['p95']:6.1f} | max {s['max']:6.1f}")
    print(f"  => {result['fps']} fps (processing only; excludes video decode)")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{device_name()}_{backend_label()}".lower().replace(" ", "_")
    path = os.path.join(args.out_dir, f"benchmark_{tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
