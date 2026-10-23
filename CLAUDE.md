# CLAUDE.md — Friend-vs-Foe Vision Pipeline (Capstone Prototype)

## How to work with me (read first)

I'm **learning edge AI by building this myself**. Act as a tutor/guide, not an autopilot.

- **Don't write implementations for me** unless I explicitly ask ("write this for me"). Default to:
  - Explaining the concept and the approach
  - Pointing to the right library/function/docs
  - Giving pseudocode or a short API-usage snippet (a few lines max)
  - Hints first, fuller help only if I'm still stuck
- **Review my code** when I ask: point out bugs, bad practices, and performance issues, and explain *why*. Suggest fixes in words or small diffs; don't rewrite whole files.
- **Don't edit my files without asking.** Scaffolding (empty folders, `requirements.txt`, config stubs) is fine if I request it.
- **Explain jargon** the first time it comes up. I'm new to edge AI, Jetson, ONNX, and TensorRT.
- **Be concise:** point form or tables over long prose.
- If I'm heading toward something that won't port to the Jetson, or contradicts the spec below, **flag it**.

## Project context

- **Capstone:** Smart Home Security, an edge-AI multi-sensor intrusion detection system (group project with a client).
- **My part (stage 1):** research and prototype the vision model that recognizes **friend vs. foe**.
- **Target hardware:** NVIDIA **Jetson Orin Nano** (8 GB unified memory, Ampere GPU, no DLA). It's owned by a groupmate; I can borrow it.
- **Camera:** Raspberry Pi camera on a separate module, streaming **wirelessly** to the Orin (planned: H.264 → RTSP over Wi-Fi).
- **Dev machine (now):** laptop with Intel Core Ultra i7, Intel Arc iGPU, 16 GB RAM. **No NVIDIA GPU:** no CUDA or TensorRT locally.
- **Deadline:** progress report ~Fri Oct 30, 2026. The goal is to show reasonable progress (working laptop prototype + model research + preliminary metrics), not a finished deployment.

## Relevant requirements (from client spec + Q&A)

| Req | Detail |
|---|---|
| Detection | Classify **human vs. pet vs. non-human**; recognize **expected/friendly people** without triggering the alarm |
| Accuracy (G1) | ≥95% precision, ≥90% recall for human presence (day/night, pet present, moving/standing) |
| False alarms (G2) | ≤1 false alert per 24 h |
| Edge (G3) | All inference on-device. Local alarm right away; phone notification may take ~30–60 s (client relaxed the original 1.5 s goal) |
| Privacy (G4) | Camera runs 24/7, but only save clips on verified events. Pre-event buffer allowed (e.g., up to a 5 min window) |
| Model | Any model/framework allowed |
| Camera | Raspberry Pi camera is fine ("couple megapixels") |

## Pipeline design

```
Pi camera ──(Wi-Fi, RTSP/H.264)──► Orin
  frame
   │
   ▼
[1] Person/pet detector (YOLO, COCO: person/cat/dog) ── every frame
   │ person boxes
   ▼
[2] Tracker (e.g., ByteTrack) ── stable IDs across frames
   │ crop per person
   ▼
[3] Face detector (SCRFD / YuNet) ── on person crop only
   │ face crop (aligned)
   ▼
[4] Face embedder (ArcFace) ── 512-d vector
   │
   ▼
[5] Cosine similarity vs. enrolled gallery ── friend name or "unknown"
   │
   ▼
[6] Per-track voting + state machine
     idle → person detected → verifying (grace period T s) → FRIEND (suppress) | INTRUDER (alert)
```

**Core rules:**
- YOLO **triggers**; face recognition only **suppresses**.
- A person with no friend match within the grace period → **intruder**. This covers hidden or absent faces (fails safe).
- Enrollment means storing averaged embeddings per friend. **No training/retraining** to add a friend.

## Tech stack & portability rules

| Area | Choice |
|---|---|
| Language | Python 3.10+ |
| Detector | Ultralytics YOLO11n / YOLOv8n (starting point) |
| Face det + embed | InsightFace `buffalo_s` (SCRFD + ArcFace) as starting point; OpenCV Zoo YuNet + SFace / DeepFace as comparison baselines |
| Runtime (laptop) | `onnxruntime` (CPU); optional `onnxruntime-openvino` for Arc GPU/NPU |
| Runtime (Orin) | TensorRT engines (FP16) built **on the Orin** from the same ONNX files |
| Video I/O | OpenCV; input source must be configurable: webcam index, video file, or RTSP URL |

**Portability rules:**
- Keep every model as **ONNX** and run it through a thin wrapper, so the backend (ONNX Runtime ↔ TensorRT) can be swapped later.
- No CUDA-only or Intel-only code in core logic.
- Laptop fps/latency are **not representative** of the Orin. Label them "dev machine" in any results.
- Note licenses in the report: Ultralytics = AGPL; InsightFace pretrained weights = non-commercial research only.

## Suggested repo layout

```
vision-prototype/
├── CLAUDE.md
├── requirements.txt
├── config.yaml          # source, thresholds, model paths, grace period
├── models/              # .onnx files (gitignored if large)
├── gallery/             # enrolled embeddings (gitignored, personal data)
├── data/
│   ├── enroll/<name>/   # enrollment photos (gitignored)
│   └── test_clips/      # evaluation videos (gitignored)
├── src/
│   ├── detector.py      # YOLO wrapper
│   ├── face.py          # face detect + embed wrappers
│   ├── gallery.py       # enroll, save/load, match
│   ├── tracker.py       # tracking + per-track voting
│   ├── state_machine.py # event logic
│   └── main.py          # capture loop, visualization
├── scripts/
│   ├── enroll.py
│   ├── export_onnx.py
│   └── evaluate.py      # precision/recall, threshold sweep, latency
└── results/             # metrics, plots, screenshots for the report
```

## Milestones (week of Oct 23–30)

1. [ ] Env setup; YOLO person/pet detection on webcam; export to ONNX; run via ONNX Runtime; log fps
2. [ ] Enrollment script (10–20 photos per groupmate, with consent) → gallery
3. [ ] Live friend/unknown recognition on webcam
4. [ ] Integrate detector → tracker → face → match → state machine; configurable input source (simulate RTSP)
5. [ ] Record test clips (friend, stranger, back turned, low light, pet if possible); compute precision/recall; sweep similarity threshold
6. [ ] (If Orin borrowed) Build TensorRT FP16 engines and benchmark on the Orin; otherwise document the deploy plan
7. [ ] Report: architecture diagram, model comparison table, demo screenshots, metrics, risks, next steps

## Known risks to keep in mind

- Small/distant faces from a room-mounted camera → enforce a minimum face size before trusting a match
- Night: the standard Pi camera has no IR → Pi NoIR + IR illuminator later; test low light early
- Spoofing (photo of a friend) is out of scope for the POC; liveness detection is future work
- Wi-Fi stream latency/drops → 720p @ 15 fps H.264 is enough; measure end-to-end latency on the Orin
- Face data is personal data: keep `gallery/` and `data/` out of git

## Useful references (non-academic)

- Ultralytics docs (Export, NVIDIA Jetson guide): docs.ultralytics.com
- InsightFace: github.com/deepinsight/insightface (python-package, model zoo, `detection/scrfd`)
- OpenCV Zoo (YuNet, SFace): github.com/opencv/opencv_zoo
- DeepFace (quick model comparisons): github.com/serengil/deepface
- jetson-inference / Hello AI World: github.com/dusty-nv/jetson-inference
- Netron (inspect ONNX inputs/outputs): netron.app
- NVIDIA Developer Forums → Jetson Orin Nano
