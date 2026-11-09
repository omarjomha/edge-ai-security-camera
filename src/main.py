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
from src.state_machine import FRIEND, GUEST, INTRUDER, StateMachine
from src.runtime import backend_label, configure, describe, device_name
from src.tracker import PERSON, Track, Tracker, assign_faces

# Label colours (BGR)
GREEN, RED, GREY, YELLOW, ORANGE = (0, 200, 0), (0, 0, 255), (160, 160, 160), (0, 220, 255), (0, 140, 255)


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
        elif p.state == GUEST:
            label, colour = f"#{t.track_id} GUEST (with {p.host})", ORANGE
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


def parse_source(source):
    """"0" (command line / config) -> webcam index 0; anything else unchanged."""
    return int(source) if isinstance(source, str) and source.isdigit() else source


def is_file(source) -> bool:
    return isinstance(source, str) and os.path.isfile(source)


def open_capture(source, cfg: dict) -> cv2.VideoCapture | None:
    """Open a webcam index, video file, rtsp:// URL or GStreamer pipeline. None if it fails."""
    if isinstance(source, str) and "!" in source:
        # GStreamer pipeline (Orin hardware decode, e.g. "rtspsrc ... ! nvv4l2decoder ! ...")
        cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    elif isinstance(source, str) and source.startswith("rtsp://"):
        # TCP resends lost Wi-Fi packets (clean frames, a little more delay); UDP shows
        # losses as smeared frames. Must be set before the capture is created.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{cfg['rtsp_transport']}"
        timeout_ms = int(cfg["stream_timeout_s"] * 1000)  # default would hang ~30 s on a dead stream
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
                                                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms])
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        return None
    if not is_file(source):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep only the newest frame (not all backends honour it)
    return cap


def connect(source, cfg: dict, start: float) -> cv2.VideoCapture:
    """Files: open once or fail. Live sources: retry until the camera is reachable."""
    while True:
        cap = open_capture(source, cfg)
        if cap is not None:
            return cap
        if is_file(source):
            raise RuntimeError(f"Could not open video file: {source!r}")
        print(f"[{time.perf_counter() - start:6.1f}s] cannot reach {source!r}, "
              f"retrying in {cfg['reconnect_delay_s']} s")
        time.sleep(cfg["reconnect_delay_s"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", help="override config: webcam index, video path, rtsp:// URL "
                                         "or GStreamer pipeline")
    parser.add_argument("--no-show", action="store_true", help="no window (benchmarking / SSH on the Orin)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    source = args.source if args.source is not None else cfg["source"]

    configure(cfg)                          # pick cpu / cuda / tensorrt before loading models
    if cfg.get("backend") == "tensorrt":
        print("tensorrt: the first run builds engines (can take minutes); later runs load the cache")
    detector = Detector(cfg["detector_model"], conf=cfg["detector_conf"])
    face_det = FaceDetector(cfg["face_det_model"], conf=cfg["face_conf"],
                            input_size=cfg["face_det_size"])
    embedder = FaceEmbedder(cfg["face_embed_model"])
    if not os.path.exists(cfg["gallery_path"]):
        raise SystemExit(f"No gallery at {cfg['gallery_path']}: run  python -m scripts.enroll  first")
    gallery = load_gallery(cfg["gallery_path"])   # loaded once; no hot-reload yet
    tracker = Tracker(cfg["track_high_conf"], cfg["track_new_conf"], cfg["track_match_iou"],
                      cfg["track_max_misses"], cfg["track_min_hits"], cfg["vote_window"])
    sm = StateMachine(cfg["grace_period_s"], cfg["vote_min_matches"], cfg["allow_accompanied_guests"], cfg["guest_window_s"], cfg["guest_arrival_gap_s"])
    print(f"backend {backend_label()} | YOLO {describe(detector.session)} | "
          f"SCRFD {describe(face_det.session)} | ArcFace {describe(embedder.session)}")
    print(f"gallery: {sorted(gallery)} | threshold {cfg['similarity_threshold']} | "
          f"grace {cfg['grace_period_s']} s")
    source = parse_source(source)
    from_file = is_file(source)
    start = time.perf_counter()
    cap = connect(source, cfg, start)

    frames, detect_total, face_total, recog_total = 0, 0.0, 0.0, 0.0
    fps = 0.0
    while True:
        loop_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            if from_file:                   # end of the video: done
                break
            # Live source dropped (Wi-Fi, camera restart, tampering?): report it and keep
            # retrying. Whether a lost camera raises an alert is a system-integration decision.
            lost_at = time.perf_counter()
            print(f"[{lost_at - start:6.1f}s] STREAM LOST: reconnecting")
            cap.release()
            cap = connect(source, cfg, start)
            print(f"[{time.perf_counter() - start:6.1f}s] stream reconnected after "
                  f"{time.perf_counter() - lost_at:.1f} s")
            # Fail safe: we can't know who moved during the blackout, so drop all tracks.
            # Everyone in view is re-verified (state machine reports them as "left").
            tracker.reset()
            print(f"[{time.perf_counter() - start:6.1f}s] tracks reset: everyone in view is re-verified")
            continue

        # Video files: use the video's own clock (files can run faster than real time).
        # Live sources: wall clock.
        now = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if from_file
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
                        f"recog {recog_ms:.0f} ms ({device_name()}, {backend_label()})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("friend vs foe", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # waitKey also lets the window redraw
                break

    cap.release()
    cv2.destroyAllWindows()

    if frames:
        elapsed = time.perf_counter() - start
        print(f"[{device_name()}, {backend_label()}] {frames} frames | avg detect {detect_total / frames:.1f} ms | "
              f"avg face {face_total / frames:.1f} ms | avg recog {recog_total / frames:.1f} ms | "
              f"avg loop {frames / elapsed:.1f} fps")


if __name__ == "__main__":
    main()
