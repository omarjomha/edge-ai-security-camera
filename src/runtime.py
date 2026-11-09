"""Model sessions shared by every model (YOLO, SCRFD, ArcFace): the one backend swap point.

A session = one loaded, ready-to-run model. Creating it is slow (on the Orin the first
time: TensorRT builds an engine, minutes); session.run() per frame is fast.

The backend is chosen here, once, for all models:
  cpu      : ONNX Runtime on the CPU (laptop / any machine)
  cuda     : ONNX Runtime's CUDA kernels (needs an onnxruntime-gpu built for this GPU)
  tensorrt : native TensorRT (JetPack's own TensorRT + cuda-python): FP16 engines built
             from the .onnx files on first use, cached on disk. Used on the Orin, because
             no onnxruntime-gpu build supports JetPack 7 / CUDA 13 on the Orin's GPU.
"""
import os
from collections import namedtuple

import numpy as np
import onnxruntime as ort

# Set once at startup by configure(); scripts that never call it stay on the CPU
_settings = {"backend": "cpu", "fp16": True, "cache_dir": "models/trt_cache"}

ORT_PROVIDER = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}
BACKENDS = ("cpu", "cuda", "tensorrt")


def configure(cfg: dict) -> None:
    """Pick the backend from config.yaml. Fails loudly if it isn't installed: a silent
    fallback to the CPU would make benchmarks wrong."""
    backend = cfg.get("backend", "cpu")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {list(BACKENDS)}, got {backend!r}")
    if backend == "tensorrt":
        try:
            import tensorrt  # noqa: F401  (ships with JetPack)
            _cudart()
        except ImportError as e:
            raise SystemExit(f"backend 'tensorrt' needs JetPack's tensorrt module and "
                             f"cuda-python ({e}). See docs/orin_setup.md.")
    elif ORT_PROVIDER[backend] not in ort.get_available_providers():
        raise SystemExit(f"backend {backend!r} needs {ORT_PROVIDER[backend]}, but this "
                         f"onnxruntime only has {ort.get_available_providers()}.")
    _settings.update(backend=backend, fp16=cfg.get("trt_fp16", True),
                     cache_dir=cfg.get("trt_cache_dir", "models/trt_cache"))


def session_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    # By default ORT worker threads busy-wait ("spin") after each run. With several
    # models running one after another, the idle spinners of one model steal CPU from
    # the next: measured YOLO + SCRFD 280 ms -> 40 ms on the dev machine with this off.
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return opts


def make_session(model_path: str, cache_tag: str = "", input_shape: tuple | None = None):
    """Load one model on the configured backend.

    cache_tag:   distinguishes engines of the same model built for different sizes
    input_shape: fixed input shape for models with variable dims (TensorRT needs one)
    """
    backend = _settings["backend"]
    if backend == "tensorrt":
        return TrtSession(model_path, cache_tag, input_shape)
    providers = [ORT_PROVIDER[backend]] + (["CPUExecutionProvider"] if backend == "cuda" else [])
    session = ort.InferenceSession(model_path, session_options(), providers=providers)
    if session.get_providers()[0] != ORT_PROVIDER[backend]:
        raise RuntimeError(f"{model_path}: asked for {backend} but got {session.get_providers()[0]}")
    return session


# ---------------------------------------------------------------------------------------
# Native TensorRT: a small stand-in for ort.InferenceSession (get_inputs / get_outputs /
# run / get_providers), so the model classes don't care which backend they run on.
# ---------------------------------------------------------------------------------------
IOInfo = namedtuple("IOInfo", ["name", "shape"])
_trt_logger = None      # TensorRT expects one logger per process, shared by all engines


def _logger(trt):
    global _trt_logger
    if _trt_logger is None:
        _trt_logger = trt.Logger(trt.Logger.WARNING)
    return _trt_logger


def _cudart():
    """cuda-python's CUDA runtime API (module path changed between versions)."""
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart
    return cudart


def _check(result):
    """cuda-python calls return (error, *values): raise on error, return the values."""
    err, *values = result
    if int(err) != 0:
        raise RuntimeError(f"CUDA error: {err}")
    return values[0] if len(values) == 1 else values


class TrtSession:
    def __init__(self, model_path: str, cache_tag: str = "", input_shape: tuple | None = None):
        import tensorrt as trt
        self.cudart = _cudart()
        logger = _logger(trt)
        precision = "fp16" if _settings["fp16"] else "fp32"
        stem = os.path.splitext(os.path.basename(model_path))[0]
        # Engines are tied to this GPU + TensorRT version: the version is in the file name
        engine_path = os.path.join(_settings["cache_dir"],
                                   f"{stem}{cache_tag}_{precision}_trt{trt.__version__}.engine")
        if not os.path.exists(engine_path):
            print(f"building TensorRT {precision} engine for {stem}{cache_tag} "
                  f"(first run only, can take minutes)...", flush=True)
            self._build(trt, logger, model_path, engine_path, input_shape)

        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = _check(self.cudart.cudaStreamCreate())

        # One fixed-size GPU buffer per input/output, allocated once
        self.inputs, self.outputs, self.device = [], [], {}
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        for name in names:      # inputs first: their shapes fix the output shapes
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = tuple(self.engine.get_tensor_shape(name))
                if -1 in shape:
                    shape = tuple(input_shape)
                    self.context.set_input_shape(name, shape)
                self.inputs.append(IOInfo(name, list(shape)))
        self.host_out = {}
        for name in names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self.device[name] = _check(self.cudart.cudaMalloc(nbytes))
            self.context.set_tensor_address(name, int(self.device[name]))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.outputs.append(IOInfo(name, list(shape)))
                self.host_out[name] = np.empty(shape, dtype=dtype)

    @staticmethod
    def _build(trt, logger, model_path: str, engine_path: str, input_shape: tuple | None) -> None:
        """.onnx -> optimized TensorRT engine (FP16 if configured), saved to disk."""
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        with open(model_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(f"TensorRT can't parse {model_path}: {errors}")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)   # 1 GB scratch
        if _settings["fp16"]:
            config.set_flag(trt.BuilderFlag.FP16)    # half precision: ~2x faster
        # Variable input dims (SCRFD's size, ArcFace's batch) need one concrete shape
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            shape = tuple(inp.shape)
            if -1 in shape:
                if input_shape is None:
                    raise ValueError(f"{model_path}: input {inp.name} has variable dims "
                                     f"{shape}; pass input_shape")
                shape = tuple(input_shape)
            profile.set_shape(inp.name, shape, shape, shape)
        config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"TensorRT engine build failed for {model_path}")
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized)

    def get_inputs(self) -> list[IOInfo]:
        return self.inputs

    def get_outputs(self) -> list[IOInfo]:
        return self.outputs

    def get_providers(self) -> list[str]:
        return ["TensorRT"]

    def run(self, output_names, feed: dict) -> list[np.ndarray]:
        """Same call as ort.InferenceSession.run: copy in -> execute on GPU -> copy out."""
        cu = self.cudart
        h2d, d2h = cu.cudaMemcpyKind.cudaMemcpyHostToDevice, cu.cudaMemcpyKind.cudaMemcpyDeviceToHost
        for name, array in feed.items():
            array = np.ascontiguousarray(array, dtype=np.float32)
            _check(cu.cudaMemcpyAsync(self.device[name], array.ctypes.data, array.nbytes,
                                      h2d, self.stream))
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("TensorRT execution failed")
        names = output_names or [o.name for o in self.outputs]
        for name in names:
            out = self.host_out[name]
            _check(cu.cudaMemcpyAsync(out.ctypes.data, self.device[name], out.nbytes,
                                      d2h, self.stream))
        _check(cu.cudaStreamSynchronize(self.stream))   # wait until the copies are done
        return [self.host_out[name].copy() for name in names]


def describe(session) -> str:
    """Which provider this session actually uses, e.g. for the startup log."""
    return session.get_providers()[0].replace("ExecutionProvider", "")


def device_name() -> str:
    """'NVIDIA Jetson Orin Nano ...' on a Jetson, else 'dev machine' (for labelling results)."""
    try:
        with open("/proc/device-tree/model") as f:      # exists on Jetson boards
            return f.read().strip("\x00 \n")
    except OSError:
        return "dev machine"


def backend_label() -> str:
    """Short label for logs/benchmarks, e.g. 'cpu' or 'tensorrt fp16'."""
    b = _settings["backend"]
    return f"{b} fp16" if b == "tensorrt" and _settings["fp16"] else b
