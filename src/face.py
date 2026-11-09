"""Live-pipeline face code: detect (SCRFD), align, embed (ArcFace).

Quick-check tools for this module live in scripts/face_check.py."""
import cv2
import numpy as np

from src.runtime import make_session

SCRFD_STRIDES = (8, 16, 32)   # three grids: fine (small faces), medium, coarse (big faces)
SCRFD_ANCHORS = 2             # anchors per grid cell (same centre)


class FaceDetector:
    """SCRFD-500M face detector: boxes + 5 landmarks. Plain ONNX Runtime via src/runtime.py,
    with our own pre/post-processing (no insightface at runtime)."""

    def __init__(self, model_path: str, conf: float = 0.5, input_size: int = 640,
                 nms_iou: float = 0.4):
        self.session = make_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        # 9 outputs: scores for strides 8/16/32, then box distances x3, then landmarks x3
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = input_size  # the model accepts any size; we fix it (TensorRT-friendly)
        self.conf = conf
        self.nms_iou = nms_iou
        # Grid points never change for a fixed input size: compute them once
        self.anchor_centers = {s: self._anchor_centers(s) for s in SCRFD_STRIDES}

    def _anchor_centers(self, stride: int) -> np.ndarray:
        """(cells * 2, 2) pixel positions of every anchor for one stride, row by row."""
        n = self.input_size // stride
        ys, xs = np.mgrid[:n, :n]
        centers = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32) * stride
        return np.repeat(centers, SCRFD_ANCHORS, axis=0)

    def preprocess(self, img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Resize (keep aspect ratio) into the top-left of a black square -> (1, 3, S, S)."""
        h, w = img_bgr.shape[:2]
        s = self.input_size
        if h > w:                         # same rounding as SCRFD's reference code
            new_h, new_w = s, int(s * w / h)
        else:
            new_h, new_w = int(s * h / w), s
        canvas = np.zeros((s, s, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(img_bgr, (new_w, new_h))
        # (pixel - 127.5) / 128, BGR -> RGB, HWC -> NCHW
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (s, s), (127.5, 127.5, 127.5), swapRB=True)
        return blob, new_h / h

    def detect(self, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """BGR image -> (boxes, landmarks).

        boxes:     (N, 5)    x1, y1, x2, y2, score   in image pixels
        landmarks: (N, 5, 2) (x, y) for left eye, right eye, nose, left mouth, right mouth
        """
        blob, scale = self.preprocess(img_bgr)
        outs = self.session.run(self.output_names, {self.input_name: blob})

        all_boxes, all_scores, all_kps = [], [], []
        for i, stride in enumerate(SCRFD_STRIDES):
            scores = outs[i][:, 0]
            keep = scores >= self.conf
            centers = self.anchor_centers[stride][keep]
            # The model predicts distances (in grid units) from each anchor point to the
            # box's left/top/right/bottom edges, and offsets to the 5 landmarks
            dist = outs[i + 3][keep] * stride
            kps = outs[i + 6][keep].reshape(-1, 5, 2) * stride
            all_boxes.append(np.column_stack([centers - dist[:, :2], centers + dist[:, 2:]]))
            all_scores.append(scores[keep])
            all_kps.append(centers[:, None, :] + kps)

        boxes = np.concatenate(all_boxes) / scale      # undo the resize
        scores = np.concatenate(all_scores)
        kps = np.concatenate(all_kps) / scale
        if len(boxes) == 0:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)

        xywh = np.column_stack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]])
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), self.conf, self.nms_iou)
        idx = np.asarray(idx, dtype=int).flatten()
        idx = idx[np.argsort(-scores[idx])]            # highest score first
        dets = np.column_stack([boxes[idx], scores[idx]]).astype(np.float32)
        return dets, kps[idx].astype(np.float32)


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


# Where ArcFace expects the 5 landmarks to be in its 112x112 input
# (left eye, right eye, nose, left mouth, right mouth). Standard insightface values.
ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963],
                             [73.5318, 51.5014],
                             [56.0252, 71.7366],
                             [41.5493, 92.3655],
                             [70.7299, 92.2041]], dtype=np.float32)
ALIGNED_SIZE = 112


def align_face(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Rotate/scale/shift one face so its 5 landmarks land on ARCFACE_TEMPLATE.

    landmarks: (5, 2) for this face, in img pixels. Returns a 112x112 BGR image.
    """
    # "Partial" affine = similarity transform: rotation + uniform scale + shift, no shear,
    # so the face isn't distorted. Least-squares fit over all 5 point pairs.
    matrix, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), ARCFACE_TEMPLATE,
                                            method=cv2.LMEDS)  # robust to one bad landmark
    return cv2.warpAffine(img, matrix, (ALIGNED_SIZE, ALIGNED_SIZE), borderValue=0)


class FaceEmbedder:
    """ArcFace: aligned 112x112 face -> 512-d unit vector. Plain ONNX Runtime, like
    src/detector.py, so the backend can be swapped for TensorRT on the Orin."""

    def __init__(self, model_path: str):
        self.session = make_session(model_path)
        self.input_name = self.session.get_inputs()[0].name  # "input.1"

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Aligned face (112, 112, 3) uint8 BGR -> (512,) float32 with length 1."""
        img = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)        # trained on RGB
        img = (img.astype(np.float32) - 127.5) / 127.5            # 0..255 -> -1..1
        blob = img.transpose(2, 0, 1)[np.newaxis]                 # HWC -> (1, 3, 112, 112)
        vec = self.session.run(None, {self.input_name: np.ascontiguousarray(blob)})[0][0]
        return vec / np.linalg.norm(vec)  # L2-normalize: only the direction matters
