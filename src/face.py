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


def detect_in_persons(face_det: FaceDetector, frame: np.ndarray, dets: np.ndarray,
                      margin: float = 0.1, iou_threshold: float = 0.4
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Run the face detector on each YOLO person box only (not the whole frame).

    dets: (N, 6) from Detector.detect -> x1, y1, x2, y2, score, class_id
    Returns (boxes, landmarks) like FaceDetector.detect, but in full-frame pixels.
    """
    frame_h, frame_w = frame.shape[:2]
    all_boxes, all_landmarks = [], []
    for x1, y1, x2, y2, _, cls in dets:
        if int(cls) != 0:                       # COCO class 0 = person; skip cats/dogs
            continue
        # Grow the box a little so a tight box doesn't cut off the top of the head
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        cx1, cy1 = int(max(x1 - mx, 0)), int(max(y1 - my, 0))
        cx2, cy2 = int(min(x2 + mx, frame_w)), int(min(y2 + my, frame_h))
        crop = frame[cy1:cy2, cx1:cx2]          # numpy slice: a view, no copy
        if crop.size == 0:
            continue

        boxes, landmarks = face_det.detect(crop)
        # Crop pixels -> frame pixels: shift by the crop's top-left corner
        boxes = boxes.copy()
        boxes[:, [0, 2]] += cx1
        boxes[:, [1, 3]] += cy1
        landmarks = landmarks + np.array([cx1, cy1], dtype=np.float32)
        all_boxes.append(boxes)
        all_landmarks.append(landmarks)

    if not all_boxes:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32)
    boxes, landmarks = np.concatenate(all_boxes), np.concatenate(all_landmarks)

    # Overlapping person boxes (people side by side) mean one face can be found in
    # several crops -> NMS again across all crops to keep one box per face
    xywh = np.column_stack([boxes[:, :2], boxes[:, 2:4] - boxes[:, :2]])
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), boxes[:, 4].tolist(), 0.0, iou_threshold)
    idx = np.asarray(idx, dtype=int).flatten()
    return boxes[idx], landmarks[idx]


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
