"""YOLO detector quick-check on a still image (dev tool, not part of the live pipeline).

Usage: python -m scripts.detector_check path/to/image.jpg [--conf 0.25]
"""
import argparse

import cv2

from src.detector import CLASSES, Detector, draw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--save", default="results/detections.jpg")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    det = Detector("models/yolo11n.onnx", conf=args.conf)
    dets = det.detect(frame)

    print(f"{len(dets)} detections (conf >= {args.conf}):")
    for x1, y1, x2, y2, score, cls in dets:
        print(f"  {CLASSES[int(cls)]:6} {score:.2f}  box ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")

    cv2.imwrite(args.save, draw(frame, dets))
    print(f"saved: {args.save}")
