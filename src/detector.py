"""YOLO person/pet detector wrapper (ONNX Runtime backend)."""
import sys

import cv2
import numpy as np
import onnxruntime as ort

PAD_VALUE = 114  # grey padding, same value Ultralytics used during training


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
    def __init__(self, model_path: str):
        # CPU for now; on the Orin this whole class is swapped for a TensorRT version
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name        # "images"
        self.input_size = inp.shape[2]    # 640, read from the file instead of hard-coding

    def preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """OpenCV BGR frame (H, W, 3) uint8 -> model input (1, 3, 640, 640) float32 0-1."""
        img, scale, pad = letterbox(frame_bgr, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # model was trained on RGB
        img = img.transpose(2, 0, 1)                  # HWC -> CHW
        img = img[np.newaxis].astype(np.float32) / 255.0  # add batch dim, scale to 0-1
        return np.ascontiguousarray(img), scale, pad

    def infer_raw(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run the model; returns the raw (1, 84, 8400) output (not decoded yet)."""
        blob, _, _ = self.preprocess(frame_bgr)
        return self.session.run(None, {self.input_name: blob})[0]


if __name__ == "__main__":
    # Quick check: python -m src.detector path/to/image.jpg
    frame = cv2.imread(sys.argv[1])
    det = Detector("models/yolo11n.onnx")

    blob, scale, pad = det.preprocess(frame)
    print(f"frame:  {frame.shape} {frame.dtype}")
    print(f"input:  {blob.shape} {blob.dtype}  range [{blob.min():.2f}, {blob.max():.2f}]")
    print(f"scale:  {scale:.3f}  pad (x, y): {pad}")

    out = det.infer_raw(frame)
    print(f"output: {out.shape} {out.dtype}")

    # Peek: the single strongest candidate (decoding properly is the next stage)
    preds = out[0].T                      # (8400, 84): one row per candidate
    best = preds[:, 4:].max(axis=1).argmax()
    cls = preds[best, 4:].argmax()
    print(f"strongest candidate #{best}: class {cls} (0 = person), "
          f"score {preds[best, 4 + cls]:.2f}, box cx,cy,w,h = {preds[best, :4].round(1)}")
