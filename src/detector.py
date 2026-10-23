"""YOLO person/pet detector wrapper (ONNX Runtime backend)."""
import argparse

import cv2
import numpy as np
import onnxruntime as ort

PAD_VALUE = 114  # grey padding, same value Ultralytics used during training

# COCO class ids we care about; every other class (bus, car, chair...) is ignored
CLASSES = {0: "person", 15: "cat", 16: "dog"}


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio, then pad to size x size.

    Returns the padded image plus the scale and (pad_x, pad_y) we need later
    to map boxes back onto the original frame.
    """
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


class Detector:
    def __init__(self, model_path: str, conf: float = 0.5, iou: float = 0.45):
        # CPU for now; on the Orin this whole class is swapped for a TensorRT version
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name        # "images"
        self.input_size = inp.shape[2]    # 640, read from the file instead of hard-coding
        self.conf = conf                  # minimum class score to keep a candidate
        self.iou = iou                    # NMS: drop boxes overlapping a better one more than this

    def preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """OpenCV BGR frame (H, W, 3) uint8 -> model input (1, 3, 640, 640) float32 0-1."""
        img, scale, pad = letterbox(frame_bgr, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # model was trained on RGB
        img = img.transpose(2, 0, 1)                  # HWC -> CHW
        img = img[np.newaxis].astype(np.float32) / 255.0  # add batch dim, scale to 0-1
        return np.ascontiguousarray(img), scale, pad

    def postprocess(self, out: np.ndarray, scale: float, pad: tuple[int, int],
                    frame_shape: tuple[int, ...]) -> np.ndarray:
        """Raw (1, 84, 8400) output -> detections on the original frame.

        Returns an (N, 6) array, one row per detection: x1, y1, x2, y2, score, class_id
        (top-left and bottom-right corners in original-frame pixels).
        """
        preds = out[0].T                       # (8400, 84): one row per candidate
        class_scores = preds[:, 4:]            # (8400, 80)
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)

        # 1. Keep confident candidates of the classes we care about
        keep = (scores >= self.conf) & np.isin(class_ids, list(CLASSES))
        boxes, scores, class_ids = preds[keep, :4], scores[keep], class_ids[keep]
        if len(boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        # 2. Centre format (cx, cy, w, h) -> corner format (x1, y1, x2, y2)
        cx, cy, w, h = boxes.T
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

        # 3. Undo the letterbox: remove the grey bars, then un-shrink
        pad_x, pad_y = pad
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
        frame_h, frame_w = frame_shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, frame_w)  # keep boxes inside the frame
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, frame_h)

        # 4. NMS per class (a person box never suppresses a dog box).
        #    OpenCV wants (x, y, w, h) with top-left corner.
        xywh = np.column_stack([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]])
        idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), scores.tolist(), class_ids.tolist(),
                                      self.conf, self.iou)
        idx = np.asarray(idx, dtype=int).flatten()

        return np.column_stack([xyxy[idx], scores[idx], class_ids[idx]]).astype(np.float32)

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Full pipeline: frame -> (N, 6) detections [x1, y1, x2, y2, score, class_id]."""
        blob, scale, pad = self.preprocess(frame_bgr)
        out = self.session.run(None, {self.input_name: blob})[0]
        return self.postprocess(out, scale, pad, frame_bgr.shape)


def draw(frame: np.ndarray, dets: np.ndarray) -> np.ndarray:
    """Draw boxes + labels onto a copy of the frame."""
    out = frame.copy()
    for x1, y1, x2, y2, score, cls in dets:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, (0, 255, 0), 2)  # green, BGR order
        cv2.putText(out, f"{CLASSES[int(cls)]} {score:.2f}", (p1[0], max(p1[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


if __name__ == "__main__":
    # Quick check: python -m src.detector path/to/image.jpg [--conf 0.25]
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
