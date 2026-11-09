"""ONNX Runtime session setup shared by every model (YOLO, SCRFD, ArcFace).

A session = one loaded, ready-to-run model. Creating it is slow (on the Orin the first
time: TensorRT builds an engine, minutes); session.run() per frame is fast.

The backend is chosen here, once, for all models:
  cpu      : laptop / any machine
  cuda     : Orin GPU via ONNX Runtime's CUDA kernels (no engine build; fallback option)
  tensorrt : Orin GPU via TensorRT engines (fastest; FP16; engines cached on disk)
"""
import os

import onnxruntime as ort

# Set once at startup by configure(); scripts that never call it stay on the CPU
_settings = {"backend": "cpu", "fp16": True, "cache_dir": "models/trt_cache"}

REQUIRED_PROVIDER = {"cpu": "CPUExecutionProvider",
                     "cuda": "CUDAExecutionProvider",
                     "tensorrt": "TensorrtExecutionProvider"}


def configure(cfg: dict) -> None:
    """Pick the backend from config.yaml. Fails loudly if it isn't installed: otherwise
    ONNX Runtime would silently fall back to the CPU and benchmarks would be wrong."""
    backend = cfg.get("backend", "cpu")
    if backend not in REQUIRED_PROVIDER:
        raise ValueError(f"backend must be one of {list(REQUIRED_PROVIDER)}, got {backend!r}")
    available = ort.get_available_providers()
    if REQUIRED_PROVIDER[backend] not in available:
        raise SystemExit(f"backend {backend!r} needs {REQUIRED_PROVIDER[backend]}, but this "
                         f"onnxruntime only has {available}. On the Orin, install the Jetson "
                         f"onnxruntime-gpu build (see docs/orin_setup.md).")
    _settings.update(backend=backend, fp16=cfg.get("trt_fp16", True),
                     cache_dir=cfg.get("trt_cache_dir", "models/trt_cache"))


def session_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    # By default ORT worker threads busy-wait ("spin") after each run. With several
    # models running one after another, the idle spinners of one model steal CPU from
    # the next: measured YOLO + SCRFD 280 ms -> 40 ms on the dev machine with this off.
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return opts


def providers(model_path: str, cache_tag: str = "") -> list:
    """Execution providers in fallback order: ops one provider can't run go to the next."""
    backend = _settings["backend"]
    if backend == "cpu":
        return ["CPUExecutionProvider"]
    if backend == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # TensorRT: one cache folder per model + input size, so e.g. SCRFD at 640 (enrollment)
    # and 320 (live) don't overwrite each other's engine and force a rebuild
    name = os.path.splitext(os.path.basename(model_path))[0] + cache_tag
    cache = os.path.join(_settings["cache_dir"], name)
    os.makedirs(cache, exist_ok=True)
    trt_options = {
        "trt_fp16_enable": _settings["fp16"],        # half precision: ~2x faster on the GPU
        "trt_engine_cache_enable": True,             # build once (minutes), then load (seconds)
        "trt_engine_cache_path": cache,
        "trt_timing_cache_enable": True,             # also speeds up rebuilds
        "trt_timing_cache_path": cache,
    }
    return [("TensorrtExecutionProvider", trt_options), "CUDAExecutionProvider", "CPUExecutionProvider"]


def make_session(model_path: str, cache_tag: str = "") -> ort.InferenceSession:
    session = ort.InferenceSession(model_path, session_options(),
                                   providers=providers(model_path, cache_tag))
    used = session.get_providers()[0]
    if used != REQUIRED_PROVIDER[_settings["backend"]]:   # e.g. TensorRT failed to load
        raise RuntimeError(f"{model_path}: asked for {_settings['backend']} but got {used}")
    return session


def describe(session: ort.InferenceSession) -> str:
    """Which provider this session actually uses, e.g. for the startup log."""
    return session.get_providers()[0].replace("ExecutionProvider", "")


def backend_label() -> str:
    """Short label for logs/benchmarks, e.g. 'cpu' or 'tensorrt fp16'."""
    b = _settings["backend"]
    return f"{b} fp16" if b == "tensorrt" and _settings["fp16"] else b
