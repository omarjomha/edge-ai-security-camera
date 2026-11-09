"""Face detection + embedding wrappers (SCRFD/YuNet + ArcFace/SFace)."""
import argparse

import cv2
import numpy as np
from insightface.model_zoo import get_model


class FaceDetector:
    """SCRFD face detector. Portability debt: runs through insightface's own ONNX Runtime
    code; for the Orin we'd load the .onnx ourselves like src/detector.py."""

    def __init__(self, model_path: str, conf: float = 0.5, input_size: int = 640):
        # Local file only: get_model never downloads unless passed download=True
        self.model = get_model(model_path, providers=["CPUExecutionProvider"])
        # Fixed input size (letterboxed internally, like YOLO); ctx_id=-1 means CPU
        self.model.prepare(ctx_id=-1, input_size=(input_size, input_size), det_thresh=conf)

    def detect(self, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """BGR image -> (boxes, landmarks).

        boxes:     (N, 5)    x1, y1, x2, y2, score   in image pixels
        landmarks: (N, 5, 2) (x, y) for left eye, right eye, nose, left mouth, right mouth
        """
        boxes, landmarks = self.model.detect(img_bgr)  # threshold + NMS happen inside
        if landmarks is None:
            landmarks = np.zeros((0, 5, 2), dtype=np.float32)
        return boxes, landmarks


def draw_faces(img: np.ndarray, boxes: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    out = img.copy()
    for (x1, y1, x2, y2, score), pts in zip(boxes, landmarks):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 128, 0), 2)  # blue
        cv2.putText(out, f"{score:.2f} {int(x2 - x1)}px", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 128, 0), 1)
        for x, y in pts:
            cv2.circle(out, (int(x), int(y)), 2, (0, 0, 255), -1)  # red landmark dots
    return out


if __name__ == "__main__":
    # Quick check: python -m src.face path/to/image.jpg
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", default="models/scrfd_500m.onnx")
    parser.add_argument("--save", default="results/faces.jpg")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    boxes, landmarks = FaceDetector(args.model).detect(img)

    print(f"image: {img.shape}  ->  {len(boxes)} faces")
    for i, ((x1, y1, x2, y2, score), pts) in enumerate(zip(boxes, landmarks)):
        print(f"  face {i}: score {score:.2f}  size {x2 - x1:.0f}x{y2 - y1:.0f}px  "
              f"left eye {pts[0].round()}  right eye {pts[1].round()}")

    cv2.imwrite(args.save, draw_faces(img, boxes, landmarks))
    print(f"saved: {args.save}")
