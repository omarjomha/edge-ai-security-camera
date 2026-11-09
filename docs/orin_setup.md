# Running the AI Stack on the Jetson Orin Nano

> **Goal:** run the full AI stack (YOLO → tracker → SCRFD → ArcFace → votes → state machine)
> on the Orin GPU, fed by a **live H.264 stream from the laptop webcam over the network**
> (stands in for the Pi camera, which isn't ready). Measure speed and confirm FP16 accuracy.
> No sensing-hub integration or sensor triggers yet.
>
> **Status:** written on the laptop; the TensorRT path and the stream setup have not run on
> an Orin yet. Expect to adjust versions once JetPack is known (step 1).

```
Laptop                                                   Orin
──────                                                   ────
webcam ─► FFmpeg (H.264, 720p, 15 fps) ─► MediaMTX ──(LAN, RTSP/TCP :8554)──► python3 -m src.main
                                          (RTSP server)                       --source rtsp://<laptop-ip>:8554/cam
```

## How it runs on the Orin

- The **same Python code** runs on the Orin's ARM CPU (`python3 -m src.main`).
- Each model is an **ONNX Runtime session** (`src/runtime.py`). With `backend: tensorrt`,
  ONNX Runtime hands the model to **TensorRT**, which builds an **FP16 engine** for the
  Orin GPU on the first run (minutes) and caches it in `models/trt_cache/` (seconds after).
- Engines are tied to this GPU + TensorRT version: they are always built **on the Orin**.
  Only the `.onnx` files are copied.
- Stream decoding, pre/post-processing, tracker, votes and state machine run on the CPU.
- The RTSP input uses TCP, 5 s timeouts and automatic reconnect (`config.yaml`:
  `rtsp_transport`, `stream_timeout_s`, `reconnect_delay_s`).

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

```bash
# On the Orin
git clone https://github.com/omarjomha/edge-ai-security-camera.git
```

Then copy the files git ignores, from the laptop (PowerShell or Git Bash, in the repo folder):

```bash
scp models/yolo11n.onnx models/scrfd_500m.onnx models/arcface_mbf.onnx <user>@<orin-ip>:~/edge-ai-security-camera/models/
scp -r data/enroll data/test_faces data/test_clips <user>@<orin-ip>:~/edge-ai-security-camera/data/
```

- `scrfd_500m.onnx` must be the **fixed** copy (`scripts/fix_scrfd_outputs.py` already ran on
  the laptop's file). Don't take a fresh one from insightface.
- No `.pt`, no `gallery/`: the gallery is rebuilt on the Orin (step 5).
- `data/test_clips/` (synthetic clips) is only the offline fallback (end of this guide).

## 3. Python environment (Orin)

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
  inside the venv. It includes FFmpeg, so RTSP input still works.

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

## 6. Laptop: start the webcam stream

**Install once** (Windows, from a terminal):

```powershell
winget install Gyan.FFmpeg
winget install bluenviron.mediamtx
```

Open a new terminal afterwards so `ffmpeg` and `mediamtx` are on the PATH.

**a. Start the RTSP server** (leave it running in its own terminal). Start it from a folder
**outside the repo** (e.g. your home folder): it writes `auto.crt` / `auto.key` (a self-signed
certificate + private key) into the current folder, and those must never be committed.
Pass its **bundled config file**: without it MediaMTX runs with an empty configuration and
rejects the stream (`path 'cam' is not configured`, FFmpeg: `400 Bad Request`).

```powershell
cd ~
$mtx = "$env:LOCALAPPDATA\Microsoft\WinGet\Packagesluenviron.mediamtx_Microsoft.Winget.Source_8wekyb3d8bbwe"
& "$mtx\mediamtx.exe" "$mtx\mediamtx.yml"
```

It should print `[RTSP] started with listeners on :8554`. Start FFmpeg (step c) only after
that line. If Windows asks about network access, allow **private networks**.

**b. Find the webcam's name:**

```powershell
ffmpeg -hide_banner -list_devices true -f dshow -i dummy
```

Look for a `(video)` line. On Omar's laptop: `"USB2.0 FHD UVC WebCam"`. To see which sizes and
frame rates it supports: `ffmpeg -hide_banner -list_options true -f dshow -i video="<name>"`.

**c. Publish the webcam as H.264, 720p, 15 fps** (the planned Pi settings), in a second terminal:

```powershell
ffmpeg -hide_banner -loglevel warning -f dshow -video_size 1280x720 -framerate 30 -i video="USB2.0 FHD UVC WebCam" `
  -r 15 -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g 15 -b:v 2M `
  -f rtsp -rtsp_transport tcp rtsp://localhost:8554/cam
```

- Capture at the webcam's native 30 fps (many webcams, including this one, only offer 720p at
  30), then `-r 15` outputs 15 fps.
- `ultrafast` + `zerolatency`: minimal encoding delay. `-g 15`: a full keyframe every second,
  so a (re)connecting reader starts quickly.
- Check the stream (optional): `ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height,avg_frame_rate -of compact rtsp://localhost:8554/cam`
  should print `h264`, `1280`, `720`, `15/1`.

**d. Check it on the laptop first** (proves the stream works before the network is involved):

```powershell
python -m src.main --source rtsp://localhost:8554/cam
```

You should see yourself labelled FRIEND. Close it with `q`. (Verified on the laptop:
FRIEND after ~3 s including connection time; laptop CPU kept up at 15.6 fps.)

**e. Let the Orin in.** Allow inbound TCP 8554 (PowerShell **as administrator**):

```powershell
New-NetFirewallRule -DisplayName "RTSP 8554 (Orin test)" -Direction Inbound -Protocol TCP -LocalPort 8554 -Action Allow -Profile Private
```

Find the laptop's IP with `ipconfig` (IPv4 address of the Wi-Fi/Ethernet adapter). The
network must be marked **Private** in Windows, and both devices must be on the same network.
Campus Wi-Fi often blocks device-to-device traffic: a phone hotspot or home router is safer.
Remove the rule after testing:
`Remove-NetFirewallRule -DisplayName "RTSP 8554 (Orin test)"`.

## 7. Orin: run the pipeline on the stream

```bash
nc -zv <laptop-ip> 8554          # must say "succeeded"; if not: firewall / network (step 6e)
python3 -m src.main --source rtsp://<laptop-ip>:8554/cam --no-show
```

- The first run builds YOLO and SCRFD-320 engines (minutes); ArcFace's is reused.
- Check the startup line: `backend tensorrt fp16 | YOLO Tensorrt | SCRFD Tensorrt | ArcFace Tensorrt`.
- Drop `--no-show` if the Orin has a monitor attached, to see the boxes.

Test in front of the laptop webcam and watch the events in the Orin terminal:

| Do this | Expected event |
|---|---|
| Face the webcam | `#N ... FRIEND omar` within ~0.5 s |
| Turn your back / cover your face from the start | `#N VERIFYING -> INTRUDER` after 3 s |
| Then face the webcam | `INTRUDER -> FRIEND omar (alarm cleared)` |
| Someone not enrolled (with consent) | `INTRUDER` after 3 s |
| Stop FFmpeg on the laptop (Ctrl+C) | `STREAM LOST: reconnecting`, then `cannot reach ...` retries (every 2 s while MediaMTX runs; ~7 s if MediaMTX is down too) |
| Start FFmpeg again | `stream reconnected after X s` within ~2 s, then `tracks reset`: everyone in view gets a new ID and is re-verified from VERIFYING |

## 8. Benchmark on the stream

Run each **twice** and keep the second result (the first includes engine builds). Stay in
view of the webcam; for the second run, have two people in view.

```bash
python3 -m scripts.benchmark rtsp://<laptop-ip>:8554/cam --frames 600    # 40 s at 15 fps
```

Two fps numbers are printed:
- **processing fps**: how fast the AI stack could run (1 / pipeline time per frame).
- **achieved fps**: frames actually processed per second. Capped at the stream's 15 fps;
  clearly below 15 means the Orin can't keep up and frames are being dropped.

The `read` stage is time spent waiting for / decoding the next frame. Results are saved as
`results/benchmark_<device>_<backend>.json` (no images, safe to commit). Optional
comparison: `backend: cuda`, then `backend: cpu`.

Not measured by the script: **end-to-end delay** (camera → decision), which also includes
encoding and the network. Rough check: point the webcam at a running stopwatch on a phone
and compare it with the time shown in the Orin's window.

While it runs, watch the hardware in a second Orin terminal:

```bash
tegrastats        # GR3D_FREQ = GPU load, RAM used, power (VDD_IN)
```

## 9. FP16 accuracy check

```bash
python3 -m scripts.check_backend_parity data/test_faces/*.jpg data/test_faces/*.jpeg
```

Expected: same-photo similarity ≥ ~0.99, and the person-vs-person table within ~0.01 of
the CPU values. If differences approach the margin around `similarity_threshold` (0.40),
note it, and compare with `trt_fp16: false`.

## 10. Bring the results back

```bash
# from the laptop
scp "<user>@<orin-ip>:~/edge-ai-security-camera/results/benchmark_*.json" results/
```

Record for the report: JetPack/L4T version, power mode, backend, network (Wi-Fi/Ethernet),
processing and achieved fps, per-stage latency (mean and p95) for 1 and 2 people, model load
time (first run vs cached), reconnect behaviour, parity results.

## Offline fallback: video files

If the network setup fails, the same commands work on the copied clips, e.g.
`python3 -m src.main --source data/test_clips/synthetic_E_friend_and_stranger.mp4 --no-show`
and `python3 -m scripts.benchmark data/test_clips/synthetic_group_pan.mp4` (6 people).

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `backend 'tensorrt' needs TensorrtExecutionProvider` | CPU `onnxruntime` installed instead of the Jetson GPU build (step 3) |
| `asked for tensorrt but got CUDA` | TensorRT couldn't take this model. Try `backend: cuda` to keep going; note which model |
| `numpy.core.multiarray failed to import` / `_ARRAY_API not found` | NumPy 2 installed: `pip install "numpy<2"` |
| `AttributeError: ... NMSBoxesBatched` | OpenCV older than 4.7: see the opencv-python fallback in step 3 |
| FFmpeg: `400 Bad Request`; MediaMTX: `path 'cam' is not configured` | MediaMTX started without its config file (step 6a) |
| `cannot reach 'rtsp://...'` / `nc` fails | MediaMTX or FFmpeg not running, wrong IP, firewall rule missing, network not "Private", or Wi-Fi blocks device-to-device traffic |
| Works on the laptop (`localhost`) but not from the Orin | Network problem, not code: firewall / network profile / Wi-Fi isolation (step 6e) |
| Smeared or grey frames | Packet loss over UDP: keep `rtsp_transport: tcp` |
| Achieved fps well below 15 | Orin can't keep up: check power mode / `jetson_clocks`, `tegrastats`, per-stage times |
| First run "hangs" for minutes | TensorRT is building engines. Later runs load `models/trt_cache/` in seconds |
| Engine rebuilt every run | Cache folder not writable, or the model file changed |
| Out of memory | Close the desktop/browser; run headless over SSH (`--no-show`) |
