# Running the AI Stack on the Jetson Orin Nano

> **Status (Nov 9, 2026):** steps 1–8 verified on the team's Orin with the recorded test
> clips; hub integration (step 10) active, including the annotated live view in the hub's
> camera source, tested with the laptop webcam as a temporary stand-in. Still to do: the
> **live Pi camera stream** (step 9, waiting for the Pi side).

**Board used:** Jetson Orin Nano Engineering Reference Developer Kit **Super**, JetPack
**7.2.1** (L4T R39.2.1), Ubuntu 24.04, Python 3.12.3, CUDA 13.2, TensorRT 10.16, OpenCV 4.8.0
(with FFmpeg and GStreamer), NumPy 1.26.4, power mode **MAXN_SUPER**. The team's hub
software (`~/homieos`, `ece490-hub` service, Mosquitto, kiosk browser) runs on the same
board, so all measurements below include that load.

## How it runs on the Orin

```
Pi camera ──RTSP/H.264──► python -m src.main (our pipeline, ~/edge-ai-security-camera)
                            │  YOLO, SCRFD, ArcFace: TensorRT FP16 engines on the GPU
                            │  tracker, votes, state machine: CPU
                            ├─► window on the Orin's screen (friend/foe boxes), optional
                            ├─► annotated H.264 stream on tcp://127.0.0.1:5600 (stream_out)
                            │        └─► homieos camera source → console / app live view
                            └─► ~/homieos/data/vision_state.json (every frame)
                                     │
                         homieos hub (ece490-hub) ──► fusion.is_authorized_person_present()
```

- **Why native TensorRT (not ONNX Runtime) on the GPU:** no `onnxruntime-gpu` build supports
  JetPack 7 / CUDA 13 on the Orin's GPU (compute capability 8.7). The PyPI aarch64 wheel
  installs, but fails at run time with `cudaErrorNoKernelImageForDevice`. So `backend: tensorrt`
  (`src/runtime.py`) uses **JetPack's own TensorRT** plus NVIDIA's **cuda-python** bindings:
  - Each `.onnx` file becomes an **FP16 engine** on first use, cached as
    `models/trt_cache/<model><size>_fp16_trt<version>.engine`.
  - Engines are tied to this GPU and TensorRT version. They are always built on the Orin;
    only the `.onnx` files are copied.
- ONNX Runtime (CPU) is still installed: `backend: cpu` and the FP16 accuracy check use it.

## 1. SSH access from the laptop (optional, lets Claude Code / scripts run commands)

```bash
# laptop: create a key once (no passphrase so scripts can use it)
ssh-keygen -t ed25519 -N "" -C "omar-laptop orin-test" -f ~/.ssh/id_ed25519
# Orin terminal: allow it (paste the laptop's ~/.ssh/id_ed25519.pub line)
mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo "<public key line>" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
# laptop: test
ssh jetson@jetson.internal 'cat /etc/nv_tegra_release'
```

This only adds one line to `~/.ssh/authorized_keys`: passwords, other users and SSH settings
are unchanged. Remove that line to revoke it. Note: the `jetson` account has
**passwordless sudo**, so anything logged in as `jetson` effectively has root.

## 2. Check the versions

```bash
cat /etc/nv_tegra_release; dpkg-query --show nvidia-jetpack
python3 --version
python3 -c "import cv2; print(cv2.__version__)"
python3 -c "import tensorrt; print(tensorrt.__version__)"
dpkg -l | grep -E "nvinfer |cuda-toolkit"
nvpmodel -q          # power mode (MAXN_SUPER is the fastest)
```

Needed: Python ≥3.10, OpenCV ≥4.7 (`NMSBoxesBatched`), TensorRT Python module. JetPack 7.2.1
has all of them out of the box.

## 3. Copy the project

```bash
# Orin
cd ~ && git clone https://github.com/omarjomha/edge-ai-security-camera.git
# laptop, in the repo folder: files git ignores
scp models/yolo11n.onnx models/scrfd_500m.onnx models/arcface_mbf.onnx jetson@jetson.internal:~/edge-ai-security-camera/models/
scp -r data/enroll data/test_faces data/test_clips jetson@jetson.internal:~/edge-ai-security-camera/data/
```

- `scrfd_500m.onnx` must be the **fixed** copy (`scripts/fix_scrfd_outputs.py` already ran on
  the laptop's file). Compare `md5sum models/*.onnx` on both machines.
- `data/test_faces/` holds other people's photos: delete them from the Orin after step 8.
- No `gallery/`: it is rebuilt on the Orin (step 6).

## 4. Python environment (no sudo, nothing system-wide)

```bash
cd ~/edge-ai-security-camera
python3 -m venv --system-site-packages .venv        # sees JetPack's OpenCV, NumPy, TensorRT
.venv/bin/pip install -r requirements-orin.txt      # numpy<2, pyyaml, scipy, onnxruntime (CPU), cuda-python
```

Check:

```bash
.venv/bin/python -c "import numpy, cv2, tensorrt, onnxruntime; from cuda.bindings import runtime as rt; print(numpy.__version__, cv2.__version__, tensorrt.__version__, onnxruntime.__version__, rt.cudaGetDeviceCount())"
```

Expected: NumPy 1.26.x, OpenCV 4.8.0, TensorRT 10.16.x, `(<cudaError_t.cudaSuccess: 0>, 1)`.

- **Don't** rely on `onnxruntime-gpu` from PyPI on this board (see "How it runs").
- `cuda-python` must match JetPack's CUDA **major** version (13 here):
  `pip install "cuda-python>=13,<14"`.
- Keep `numpy<2`: JetPack's OpenCV is built for NumPy 1.x.

## 5. Settings (edit `config.yaml` on the Orin only)

```yaml
backend: tensorrt
vision_state_file: "/home/jetson/homieos/data/vision_state.json"   # read by the hub (step 10)
```

Everything else stays as on the laptop, so results are comparable.

## 6. Enroll

```bash
.venv/bin/python -m scripts.enroll
```

The first run builds the SCRFD-640 + ArcFace engines. Measured: **~1.5 min**. Result:
`omar: ENROLLED`, scores 0.89 / 0.89 (same as the laptop).

## 7. First run: engine builds, then check the pipeline on clips

```bash
for c in A_friend B_stranger C_face_hidden D_hidden_then_face E_friend_and_stranger group_pan; do
  .venv/bin/python -m src.main --source data/test_clips/synthetic_$c.mp4 --no-show
done
```

- The first run builds the remaining engines. Measured: YOLO ~6.5 min, SCRFD-320 ~1.2 min,
  ArcFace ~1.4 min. Later runs load the cache in **0.2 s**.
- The startup line must read `backend tensorrt fp16 | YOLO TensorRT | SCRFD TensorRT | ArcFace TensorRT`.
- Measured events are **identical to the laptop**: friend FRIEND at 0.4 s, stranger INTRUDER at
  3.1 s, hidden face INTRUDER at 3.9 s, hidden-then-face cleared at 6.3 s, friend + stranger →
  GUEST at 3.1 s, group of 6 → all INTRUDER.
- A `WeightsContext.cpp: Empty initializer 385` warning while building SCRFD is harmless.

## 8. Benchmark and FP16 accuracy check

```bash
.venv/bin/python -m scripts.benchmark data/test_clips/synthetic_E_friend_and_stranger.mp4   # 2 people
.venv/bin/python -m scripts.benchmark data/test_clips/synthetic_group_pan.mp4               # 4–6 people
.venv/bin/python -m scripts.check_backend_parity data/test_faces/*.jpg data/test_faces/*.jpeg
```

Measured (MAXN_SUPER, hub + kiosk running, warmup excluded; JSON in `results/orin/`):

| Stage, mean ms | 2 people | ~4.3 people | Orin CPU only, 2 people |
|---|---|---|---|
| read frame | 2.7 | 2.2 | |
| detect (YOLO) | 16.1 | 16.0 | |
| face (SCRFD, per person) | 10.8 | 18.1 | |
| recog (align + ArcFace) | 5.1 | 8.1 | |
| other (tracker, votes, states) | 0.6 | 0.7 | |
| **total** | **32.6** (p95 33.4) | **43.0** (p95 54.4) | 235.3 |
| **fps** | **30.7** | **23.3** | 4.3 |

Per model call (incl. pre/post): YOLO 17.0 ms (CPU 128.7), SCRFD 4.8 ms (16.5),
ArcFace 2.5 ms (26.3).

FP16 accuracy: same-photo embedding similarity CPU vs TensorRT **0.9985–0.9998**;
person-vs-person table changes by at most **0.003** (e.g. 0.574 vs 0.577), so the 0.40
threshold is unaffected. Detections and landmarks within 0.5 px of the CPU.

Afterwards: `rm data/test_faces/<other people>.jpeg` on the Orin.

## 9. Live view on the Orin's screen (Pi camera stream)

**Pending: the Pi stream isn't ready yet.** When it is (H.264 over RTSP, ideally 1280×720,
15 fps, keyframe every 15 frames, e.g. MediaMTX on the Pi with `source: rpiCamera`):

```bash
cd ~/edge-ai-security-camera
nc -zv <pi-ip> 8554     # must succeed
DISPLAY=:1 XAUTHORITY=/run/user/2002/gdm/Xauthority \
  .venv/bin/python -m src.main --source rtsp://<pi-ip>:8554/cam
```

- The Orin's screen is the X11 session `:1` (where the kiosk runs); the two variables let a
  command started over SSH open its window there. Use `--no-show` to run headless.
- The Orin's screen is **1024×768** and the hub's kiosk browser covers it. `display_fullscreen:
  true` asks for a full-screen window, but this GNOME/mutter setup ignored the request (the
  window stayed 320×203 behind the kiosk). **Use the hub's live view instead (step 10b)**,
  which needs no window at all.
- Or set `source:` in the Orin's `config.yaml` to the Pi URL and drop `--source`.
- Stream drops: the pipeline logs `STREAM LOST`, retries, and resets all tracks after
  reconnecting (everyone is re-verified).

**Temporary stand-in used on Nov 9 (laptop webcam instead of the Pi):** on the laptop,
MediaMTX (started with its bundled `mediamtx.yml`, from a folder outside the repo) + FFmpeg
publishing the webcam to `rtsp://localhost:8554/cam`; an SSH reverse tunnel carries it to the
Orin, where it appears as `rtsp://127.0.0.1:8554/cam` (no firewall changes):

```bash
ssh -N -R 127.0.0.1:8554:127.0.0.1:8554 jetson@jetson.internal     # laptop, keep running
# Orin
.venv/bin/python -m src.main --source rtsp://127.0.0.1:8554/cam --no-show
```

## 10. Hub integration (homieos)

### 10a. Decision: fusion hook

The pipeline writes its decision to `~/homieos/data/vision_state.json` every frame
(`vision_state_file`, step 5):

```json
{"v": 1, "ts": 1794262103.87, "friend_present": true, "alarm": false,
 "people": [{"id": 1, "state": "FRIEND", "name": "omar"}, {"id": 2, "state": "GUEST", "name": "omar"}]}
```

On the Orin, the hub's placeholder `~/homieos/app/hub/fusion.py` was filled in to read it
(original kept as `fusion.py.bak-before-vision`). `is_authorized_person_present()` returns
`friend_present`, or **False** if the file is missing or older than 2 s (pipeline not
running), the same as the original placeholder. `service.py` and the hub's state machine are
unchanged.

Verified by calling the hook the same way the hub does: friend clip → True, stranger → False,
friend + guest → True, pipeline stopped → False after 2 s.

**Activate it: restart the hub** (from a terminal on the Orin; a few seconds of downtime, the
kiosk reconnects):

```bash
systemctl --user restart ece490-hub 2>/dev/null || sudo systemctl restart ece490-hub
```

Undo: `cp ~/homieos/app/hub/fusion.py.bak-before-vision ~/homieos/app/hub/fusion.py`, then
restart again. The `fusion.py` change lives only on the Orin: it should go into the homieos
repository too.

### 10b. Annotated live view in the hub's camera source

`stream_out` (Orin's `config.yaml`) encodes every annotated frame (boxes, labels, alarm
banner) with GStreamer x264 and serves it as MPEG-TS over TCP on 127.0.0.1:5600:

```yaml
stream_out: "appsrc ! videoconvert ! video/x-raw,format=I420 ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 key-int-max=15 ! h264parse config-interval=-1 ! mpegtsmux ! tcpserversink host=127.0.0.1 port=5600 sync=false"
stream_out_fps: 15
```

homieos opens it like any camera: in `~/homieos/hub.toml`, `[media] camera_url =
"tcp://127.0.0.1:5600"` (original kept as `hub.toml.bak-before-vision-camera`), then restart
the hub. Watch it at `https://jetson.internal:8443/console` → live view → **"Sensing Hub
camera"** (or in the app). Verified: homieos' own PyAV opens it in 1.5 s (H.264, 1280×720), and
the console shows the live annotated view. `stream_out` works with `--no-show` (no window).

Undo: restore `hub.toml` from the backup, restart the hub, stop the pipeline
(`pkill -f "src[.]main"`; the `[.]` stops pkill from matching its own command line).

### Open point

Open point for the team: the hub's design asks for recognition within a **1.0 s** window
after a sensor event. A recognized friend takes ~0.4 s, but our INTRUDER decision waits for
the 3 s grace period; the hook simply answers "friend present right now".

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `cudaErrorNoKernelImageForDevice` | PyPI `onnxruntime-gpu` has no code for the Orin GPU. Use `backend: tensorrt` (native) |
| `backend 'tensorrt' needs JetPack's tensorrt module and cuda-python` | venv created without `--system-site-packages`, or `cuda-python` missing (step 4) |
| `ModuleNotFoundError: No module named 'src'` | Run from the repo folder with `python -m ...`, or set `PYTHONPATH=.` |
| First run "hangs" for minutes | TensorRT is building engines (step 7). Watch `models/trt_cache/` fill up |
| Engine rebuilt every run | Cache folder not writable, model file changed, or TensorRT was updated |
| `numpy.core.multiarray failed to import` | NumPy 2 got installed: `pip install "numpy<2"` |
| Window doesn't appear on the Orin screen | Missing `DISPLAY=:1 XAUTHORITY=...`, or it's behind the kiosk browser |
| `cannot reach 'rtsp://...'` | Pi stream not running, wrong IP/path, or network blocks device-to-device traffic |
| Orin screen black | GNOME blanks and locks it after 5 min idle (`idle-delay 300`); the kernel shows `dpms=Off`. Needs keyboard/mouse input to wake and the `jetson` password to unlock |
| Clicks/keys on the remote screen do nothing | The **GL.iNet KVM's USB** (its virtual keyboard + mouse) disconnected from the Orin (Nov 9, 15:58, during full GPU load). Check `journalctl -k \| grep -i glinet`; re-seat its USB cable or reset/reboot the KVM. If it recurs under load, power the KVM separately |
| Camera source missing in the hub | `camera_url` not set, hub not restarted, or the pipeline (stream_out) isn't running |
| Hub ignores the pipeline | Hub not restarted after changing `fusion.py`, `vision_state_file` not set, or the pipeline isn't running (file older than 2 s) |
| Slower than the table above | Power mode not MAXN_SUPER, thermal throttling (`tegrastats`), or more people in view |
