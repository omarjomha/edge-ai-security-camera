# Running the AI Stack on the Jetson Orin Nano

> **Goal:** check that the AI stack (YOLO → tracker → SCRFD → ArcFace → votes → state
> machine) runs on the Orin GPU, measure its speed and confirm FP16 accuracy, using **video
> files**. No camera, streaming or sensing-hub integration yet.
>
> **Status:** written on the laptop; the TensorRT path has not run on an Orin yet. Expect
> to adjust version numbers once JetPack is known (step 1).

## How it runs on the Orin

- The **same Python code** runs on the Orin's ARM CPU (`python3 -m src.main`).
- Each model is an **ONNX Runtime session** (`src/runtime.py`). With `backend: tensorrt`,
  ONNX Runtime hands the model to **TensorRT**, which builds an **FP16 engine** for the
  Orin GPU on the first run (minutes) and caches it in `models/trt_cache/` (seconds after).
- Engines are tied to this GPU + TensorRT version: they are always built **on the Orin**.
  Only the `.onnx` files are copied.
- Pre/post-processing, tracker, votes and state machine stay on the CPU.

## 1. Check the Orin's versions

```bash
cat /etc/nv_tegra_release                    # L4T release, e.g. "R36 (release), REVISION: 4.3"
dpkg-query --show nvidia-jetpack             # JetPack version, e.g. 6.2
python3 --version                            # JetPack 6.x: 3.10
python3 -c "import cv2; print(cv2.__version__)"
dpkg -l | grep -E "nvinfer |cuda-toolkit"    # TensorRT and CUDA versions
```

| L4T | JetPack | Notes |
|---|---|---|
| R36.x | **6.x** (target) | Ubuntu 22.04, Python 3.10, CUDA 12, TensorRT 10, OpenCV 4.8 |
| R35.x | 5.x | Python 3.8 and OpenCV 4.5: **not supported as-is** (code needs Python ≥3.10 and OpenCV ≥4.7). Ask before upgrading someone else's board |

Write the versions down: they go in the results.

## 2. Copy the project to the Orin

The repo is ahead of GitHub, so either push first and `git clone` on the Orin, or copy the
folder over the network. From the laptop (PowerShell or Git Bash):

```bash
# Code (or: git clone after pushing)
scp -r edge-ai-security-camera <user>@<orin-ip>:~/
```

If you cloned instead, also copy the files git ignores:

```bash
scp models/yolo11n.onnx models/scrfd_500m.onnx models/arcface_mbf.onnx <user>@<orin-ip>:~/edge-ai-security-camera/models/
scp -r data/enroll data/test_faces data/test_clips <user>@<orin-ip>:~/edge-ai-security-camera/data/
```

- `scrfd_500m.onnx` must be the **fixed** copy (`scripts/fix_scrfd_outputs.py` already ran on
  the laptop's file). Don't take a fresh one from insightface.
- No `.pt`, no `gallery/`: the gallery is rebuilt on the Orin (step 5).
- `data/test_clips/` has synthetic clips (friend, stranger, hidden face, pair, group of 6).
  Replace or add real recordings when you have them (with consent).

## 3. Python environment

```bash
cd ~/edge-ai-security-camera
sudo apt install -y python3-venv python3-pip
python3 -m venv --system-site-packages .venv   # sees JetPack's OpenCV
source .venv/bin/activate
pip install -r requirements-orin.txt           # numpy<2, pyyaml, scipy
```

**onnxruntime-gpu (Jetson build).** The normal `pip install onnxruntime-gpu` has no ARM GPU
build. Install the wheel that matches your **JetPack / CUDA** version from NVIDIA's Jetson
sources:
- **Jetson AI Lab pip index** (JetPack 6): `pip install onnxruntime-gpu --index-url <index for your JetPack/CUDA>`. Find the current index URL on jetson-ai-lab.com.
- **Jetson Zoo** (elinux.org/Jetson_Zoo, "ONNX Runtime"): direct wheel links per JetPack.

Then check:

```bash
python3 -c "import numpy, cv2, onnxruntime as ort; print(numpy.__version__, cv2.__version__, ort.__version__); print(ort.get_available_providers())"
```

Expected: NumPy 1.x, OpenCV ≥4.7, and the providers list includes
`TensorrtExecutionProvider` and `CUDAExecutionProvider`. Don't continue until it does.

- If pip pulled in a CPU `onnxruntime` as well, remove it: `pip uninstall onnxruntime`
  (keep `onnxruntime-gpu`).
- If `import cv2` fails or OpenCV is older than 4.7: `pip install "opencv-python==4.10.0.84"`
  inside the venv. This works for video files; it lacks GStreamer, which only matters later
  for the live camera.

## 4. Settings for the Orin

Edit `config.yaml` **on the Orin only**:

```yaml
backend: tensorrt
```

Keep everything else the same as on the laptop, so results are comparable.

**Power mode** (has a big effect on speed). Use the fastest mode and record which one:

```bash
sudo nvpmodel -q                 # current mode
sudo nvpmodel -m 0               # mode 0 is usually the max mode; check the name it prints
sudo jetson_clocks               # pin clocks at max (resets on reboot)
```

Make sure the fan is running; the Orin slows itself down when hot.

## 5. Enroll on the Orin

```bash
python3 -m scripts.enroll
```

The first run builds TensorRT engines for SCRFD (640) and ArcFace: **a few minutes**, quiet
terminal. Expected: `omar: ENROLLED`, per-photo scores close to the laptop's (0.89).

## 6. Run the pipeline

```bash
python3 -m src.main --source data/test_clips/synthetic_E_friend_and_stranger.mp4 --no-show
```

- The first run builds YOLO and SCRFD-320 engines (minutes); ArcFace's is reused.
- Check the startup line: `backend tensorrt fp16 | YOLO Tensorrt | SCRFD Tensorrt | ArcFace Tensorrt`.
- Expected events (same as the laptop): `#1 ... FRIEND omar` at ~0.4 s, `#2 ... INTRUDER` at ~3.1 s.
- Try the other clips: `synthetic_C_face_hidden.mp4` must still end in INTRUDER.

## 7. Benchmark

Run each command **twice** and keep the second result (the first includes engine builds).

```bash
python3 -m scripts.benchmark data/test_clips/synthetic_E_friend_and_stranger.mp4   # 2 people
python3 -m scripts.benchmark data/test_clips/synthetic_group_pan.mp4               # 6 people
```

Optional comparison: set `backend: cuda`, then `backend: cpu`, and run again. Results are saved
as `results/benchmark_<device>_<backend>.json` (no images, safe to commit).

While it runs, watch the hardware in a second terminal:

```bash
tegrastats        # GR3D_FREQ = GPU load, RAM used, power (VDD_IN)
```

## 8. FP16 accuracy check

```bash
python3 -m scripts.check_backend_parity data/test_faces/*.jpg data/test_faces/*.jpeg
```

Expected: same-photo similarity ≥ ~0.99, and the person-vs-person table within ~0.01 of
the CPU values. If differences approach the margin around `similarity_threshold` (0.40),
note it, and compare with `trt_fp16: false`.

## 9. Bring the results back

```bash
# from the laptop
scp "<user>@<orin-ip>:~/edge-ai-security-camera/results/benchmark_*.json" results/
```

Record for the report: JetPack/L4T version, power mode, backend, fps and per-stage latency
(mean and p95) for 2 and 6 people, model load time (first run vs cached), parity results.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `backend 'tensorrt' needs TensorrtExecutionProvider` | CPU `onnxruntime` installed instead of the Jetson GPU build (step 3) |
| `asked for tensorrt but got CUDA` | TensorRT couldn't take this model. Try `backend: cuda` to keep going; note which model |
| `numpy.core.multiarray failed to import` / `_ARRAY_API not found` | NumPy 2 installed: `pip install "numpy<2"` |
| `AttributeError: ... NMSBoxesBatched` | OpenCV older than 4.7: see the opencv-python fallback in step 3 |
| First run "hangs" for minutes | TensorRT is building engines. Later runs load `models/trt_cache/` in seconds |
| Engine rebuilt every run | Cache folder not writable, or the model file changed |
| Out of memory | Close the desktop/browser; run headless over SSH (`--no-show`) |
| Much slower than expected | Power mode / `jetson_clocks` not set, or thermal throttling (check `tegrastats`) |
