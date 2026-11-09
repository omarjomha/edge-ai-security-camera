"""Make SCRFD's output shapes variable (one-time fix of the model file).

The buffalo_s SCRFD file accepts any input size but declares its outputs for a 640x640
input only (e.g. {800, 10}). At other sizes (we use 320 for person crops) ONNX Runtime
logs a shape-mismatch warning for every output on every run, which costs time, and
wrong shape metadata can also confuse TensorRT's engine builder.

Usage: python -m scripts.fix_scrfd_outputs models/scrfd_500m.onnx
(overwrites the file; the original is still in ~/.insightface/models/buffalo_s.zip)
"""
import argparse

import onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    model = onnx.load(args.model)
    for out in model.graph.output:
        # First dim = number of anchors, depends on input size -> make it a named variable
        dim = out.type.tensor_type.shape.dim[0]
        print(f"{out.name}: {dim.dim_value or dim.dim_param} -> num_anchors")
        dim.ClearField("dim_value")
        dim.dim_param = "num_anchors"
    onnx.checker.check_model(model)
    onnx.save(model, args.model)
    print(f"saved {args.model}")


if __name__ == "__main__":
    main()
