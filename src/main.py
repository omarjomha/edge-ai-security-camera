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
from src.state_machine import FRIEND, INTRUDER, StateMachine
from src.tracker import PERSON, Track, Tracker, assign_faces

# Label colours (BGR)
GREEN, RED, GREY, YELLOW = (0, 200, 0), (0, 0, 255), (160, 160, 160), (0, 220, 255)


def track_boxes(tracks: list[Track]) -> np.ndarray:
    """Tracks -> (N, 6) person "detections" so face detection runs per tracked person."""
    if not tracks:
        return np.zeros((0, 6), dtype=np.float32)
    return np.array([[*t.box, t.score, PERSON] for t in tracks], dtype=np.float32)


def identify_faces(embedder: FaceEmbedder, frame: np.ndarray, boxes: np.ndarray,
                   landmarks: np.ndarray, gallery: dict[str, np.ndarray],
                   threshold: float, min_face_px: int) -> list[tuple[str | None, float | None]]:
    """Each face -> (friend name or None, score). (None, None) = too small to judge."""
    results = []
    for box, pts in zip(boxes, landmarks):
        if box[2] - box[0] < min_face_px:   # too few pixels for a trustworthy embedding
            results.append((None, None))
            continue
        results.append(match(embedder.embed(align_face(frame, pts)), gallery, threshold))
    return results


def process_frame(frame: np.ndarray, detector: Detector, face_det: FaceDetector,
                  embedder: FaceEmbedder, gallery: dict[str, np.ndarray],
                  tracker: Tracker, sm: StateMachine, now: float, cfg: dict) -> dict:
    """One frame through the whole pipeline. Returns everything needed to draw/log it."""
    times = {}
    t0 = time.perf_counter()
    dets = detector.detect(frame)       # includes weak boxes (>= 0.1) for the tracker
    times["detect"] = (time.perf_counter() - t0) * 1000

    tracks = tracker.update(dets)       # stable person IDs; < 1 ms, not timed separately

    t0 = time.perf_counter()
    face_boxes, face_landmarks = detect_in_persons(face_det, frame, track_boxes(tracks))
    times["face"] = (time.perf_counter() - t0) * 1000  # grows with the number of people

    t0 = time.perf_counter()
    face_results = identify_faces(embedder, frame, face_boxes, face_landmarks, gallery,
                                  cfg["similarity_threshold"], cfg["min_face_px"])
    times["recog"] = (time.perf_counter() - t0) * 1000  # align + embed + match

    # Each usable face becomes one vote for the track it belongs to
    for ti, fi in assign_faces(tracks, face_boxes).items():
        name, score = face_results[fi]
        if score is not None:           # too-small faces are not evidence either way
            tracks[ti].add_vote(name)

    # All tracker tracks, including briefly lost ones: someone hiding keeps their timer
    events = sm.update(tracker.tracks, now)

    strong = dets[dets[:, 4] >= cfg["track_high_conf"]]
    pets = strong[strong[:, 5] != PERSON]
    return {"tracks": tracks, "face_boxes": face_boxes, "face_results": face_results,
            "pets": pets, "events": events, "times": times}


def draw_frame(frame: np.ndarray, out: dict, sm: StateMachine, now: float) -> np.ndarray:
    """Pets (raw boxes), people (state), faces (this frame's raw result), alarm banner."""
    img = draw(frame, out["pets"])
    for t in out["tracks"]:
        p = sm.people[t.track_id]
        if p.state == FRIEND:
            label, colour = f"#{t.track_id} FRIEND {p.friend}", GREEN
        elif p.state == INTRUDER:
            label, colour = f"#{t.track_id} INTRUDER", RED
        else:                           # VERIFYING: show the grace-period countdown
            left = max(sm.grace_period_s - (now - p.since), 0)
            label, colour = f"#{t.track_id} verifying {left:.1f}s", YELLOW
        x1, y1, x2, y2 = t.box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(img, label, (x1, max(y1 - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    for (x1, y1, x2, y2, _), (name, score) in zip(out["face_boxes"], out["face_results"]):
        if score is None:
            label, colour = "too small", GREY
        else:
            label, colour = (f"{name} {score:.2f}", GREEN) if name else (f"? {score:.2f}", RED)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
        cv2.putText(img, label, (int(x1), int(y2) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    if sm.alarm:
        cv2.rectangle(img, (0, img.shape[0] - 50), (img.shape[1], img.shape[0]), RED, -1)
        cv2.putText(img, "ALARM: INTRUDER", (15, img.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    return img


def open_source(source) -> tuple[cv2.VideoCapture, bool]:
    """Webcam index (int or "0"), video file path, or rtsp:// URL -> (capture, is_file)."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)                # "0" from the command line means webcam 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")
    return cap, isinstance(source, str) and os.path.isfile(source)


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
    tracker = Tracker(cfg["track_high_conf"], cfg["track_new_conf"], cfg["track_match_iou"],
                      cfg["track_max_misses"], cfg["track_min_hits"], cfg["vote_window"])
    sm = StateMachine(cfg["grace_period_s"], cfg["vote_min_matches"])
    print(f"gallery: {sorted(gallery)} | threshold {cfg['similarity_threshold']} | "
          f"grace {cfg['grace_period_s']} s")
    cap, is_file = open_source(source)

    frames, detect_total, face_total, recog_total = 0, 0.0, 0.0, 0.0
    fps = 0.0
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:                          # end of video file, or stream dropped
            break

        # Video files: use the video's own clock (files can run faster than real time).
        # Live sources: wall clock.
        now = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if is_file
               else time.perf_counter() - start)
        out = process_frame(frame, detector, face_det, embedder, gallery, tracker, sm, now, cfg)
        for event in out["events"]:
            print(f"[{now:6.1f}s] {event}")
        detect_ms, face_ms, recog_ms = (out["times"][k] for k in ("detect", "face", "recog"))

        frames += 1
        detect_total += detect_ms
        face_total += face_ms
        recog_total += recog_ms
        # Smoothed fps (moving average) so the number on screen doesn't flicker
        loop_fps = 1.0 / (time.perf_counter() - loop_start)
        fps = loop_fps if frames == 1 else 0.9 * fps + 0.1 * loop_fps

        if not args.no_show:
            vis = draw_frame(frame, out, sm, now)
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
