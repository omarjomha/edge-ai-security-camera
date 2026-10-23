"""Capture loop + visualization. Input: webcam index, video file, or RTSP URL."""
import argparse
import time

import cv2
import yaml

from src.detector import Detector, draw


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
    cap = open_source(source)

    frames, detect_total = 0, 0.0
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

        frames += 1
        detect_total += detect_ms
        # Smoothed fps (moving average) so the number on screen doesn't flicker
        loop_fps = 1.0 / (time.perf_counter() - loop_start)
        fps = loop_fps if frames == 1 else 0.9 * fps + 0.1 * loop_fps

        if not args.no_show:
            vis = draw(frame, dets)
            cv2.putText(vis, f"{fps:.1f} fps | detect {detect_ms:.0f} ms (dev machine)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("detector", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # waitKey also lets the window redraw
                break

    cap.release()
    cv2.destroyAllWindows()

    if frames:
        elapsed = time.perf_counter() - start
        print(f"[dev machine] {frames} frames | avg detect {detect_total / frames:.1f} ms | "
              f"avg loop {frames / elapsed:.1f} fps")


if __name__ == "__main__":
    main()
