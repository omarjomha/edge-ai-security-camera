"""ONNX Runtime session setup shared by every model (YOLO, SCRFD, ArcFace).

On the Orin these sessions are replaced by TensorRT engines; this file is where
that backend swap would happen.
"""
import onnxruntime as ort

PROVIDERS = ["CPUExecutionProvider"]


def session_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    # By default ORT worker threads busy-wait ("spin") after each run. With several
    # models running one after another, the idle spinners of one model steal CPU from
    # the next: measured YOLO + SCRFD 280 ms -> 40 ms on the dev machine with this off.
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return opts


def make_session(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(model_path, session_options(), providers=PROVIDERS)
