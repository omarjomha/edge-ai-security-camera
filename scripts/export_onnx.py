"""Export YOLO (and other) models to ONNX into models/."""
import argparse

import onnx
from ultralytics import YOLO


def export(weights: str, imgsz: int) -> str:
    model = YOLO(weights)
    # dynamic=False: fixed input shape -> simplest for ONNX Runtime and TensorRT
    # Precision stays FP32 (the default); TensorRT converts to FP16 on the Orin
    # nms=False: raw outputs; we decode boxes + NMS ourselves (portable, and we learn it)
    return model.export(format="onnx", imgsz=imgsz, dynamic=False, nms=False)


def describe(onnx_path: str) -> None:
    """Print input/output names and shapes: the 'contract' our wrapper must follow."""
    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)  # raises if the file is malformed
    for kind, tensors in (("input", m.graph.input), ("output", m.graph.output)):
        for t in tensors:
            dims = [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]
            print(f"{kind:6} {t.name}: {dims}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    path = export(args.weights, args.imgsz)
    print(f"\nExported: {path}")
    describe(path)
