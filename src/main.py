"""Capture loop + visualization. Input: webcam index, video file, or RTSP URL."""
import argparse
import os
import time

import cv2
import numpy as np
import yaml

from src.detector import Detector, draw
from src.face import FaceDetector, FaceEmbedder, align_face, detect_in_persons
from src.gallery import load_gallery, match

# Label colours (BGR)
GREEN, RED, GREY = (0, 200, 0), (0, 0, 255), (160, 160, 160)


def identify_faces(embedder: FaceEmbedder, frame: np.ndarray, boxes: np.ndarray,
                   landmarks: np.ndarray, gallery: dict[str, np.ndarray],
                   threshold: float, min_face_px: int) -> list[tuple[str, tuple]]:
    """Each face -> (label, colour). Per-frame decision only: no tracking or voting yet."""
    results = []
    for box, pts in zip(boxes, landmarks):
        if box[2] - box[0] < min_face_px:   # too few pixels for a trustworthy embedding
            results.append(("too small", GREY))
            continue
        name, score = match(embedder.embed(align_face(frame, pts)), gallery, threshold)
        if name is not None:
            results.append((f"FRIEND {name} {score:.2f}", GREEN))
        else:
            results.append((f"FOE {score:.2f}", RED))
    return results


def draw_identities(img: np.ndarray, boxes: np.ndarray, identities: list[tuple[str, tuple]]) -> np.ndarray:
    """Draw each face box in its label colour (draws on img in place)."""
    for (x1, y1, x2, y2, _), (label, colour) in zip(boxes, identities):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, 3)
        cv2.putText(img, label, (int(x1), max(int(y1) - 10, 30)),   # keep text inside the image
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
    return img


def open_source(source) -> cv2.VideoCapture:
    """Webcam index (int or "0"), video file path, or rtsp:// URL -> VideoCapture."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)                # "0" from the command line means webcam 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")
    return cap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", help="override config: webcam index, video path, or rtsp:// URL")
    parser.add_argument("--no-show", action="store_true", help="no window (benchmarking / SSH on the Orin)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    source = args.source if args.source is not None else cfg["source"]

    detector = Detector(cfg["detector_model"], conf=cfg["detector_conf"])
    face_det = FaceDetector(cfg["face_det_model"], conf=cfg["face_conf"],
                            input_size=cfg["face_det_size"])
    embedder = FaceEmbedder(cfg["face_embed_model"])
    if not os.path.exists(cfg["gallery_path"]):
        raise SystemExit(f"No gallery at {cfg['gallery_path']}: run  python -m scripts.enroll  first")
    gallery = load_gallery(cfg["gallery_path"])   # loaded once; no hot-reload yet
    print(f"gallery: {sorted(gallery)} | threshold {cfg['similarity_threshold']}")
    cap = open_source(source)

    frames, detect_total, face_total, recog_total = 0, 0.0, 0.0, 0.0
    fps = 0.0
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:                          # end of video file, or stream dropped
            break

        t0 = time.perf_counter()
        dets = detector.detect(frame)
        detect_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        face_boxes, face_landmarks = detect_in_persons(face_det, frame, dets)
        face_ms = (time.perf_counter() - t0) * 1000  # grows with the number of people

        t0 = time.perf_counter()
        identities = identify_faces(embedder, frame, face_boxes, face_landmarks, gallery,
                                    cfg["similarity_threshold"], cfg["min_face_px"])
        recog_ms = (time.perf_counter() - t0) * 1000  # align + embed + match, grows with faces

        frames += 1
        detect_total += detect_ms
        face_total += face_ms
        recog_total += recog_ms
        # Smoothed fps (moving average) so the number on screen doesn't flicker
        loop_fps = 1.0 / (time.perf_counter() - loop_start)
        fps = loop_fps if frames == 1 else 0.9 * fps + 0.1 * loop_fps

        if not args.no_show:
            vis = draw_identities(draw(frame, dets), face_boxes, identities)
            cv2.putText(vis, f"{fps:.1f} fps | detect {detect_ms:.0f} | face {face_ms:.0f} | "
                        f"recog {recog_ms:.0f} ms (dev machine)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("friend vs foe", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # waitKey also lets the window redraw
                break

    cap.release()
    cv2.destroyAllWindows()

    if frames:
        elapsed = time.perf_counter() - start
        print(f"[dev machine] {frames} frames | avg detect {detect_total / frames:.1f} ms | "
              f"avg face {face_total / frames:.1f} ms | avg recog {recog_total / frames:.1f} ms | "
              f"avg loop {frames / elapsed:.1f} fps")


if __name__ == "__main__":
    main()
