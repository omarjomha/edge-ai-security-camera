# Friend Enrollment: Design Notes

> **Status:** design notes for a later phase. The current focus is the **live AI pipeline**
> (detect → track → face → match → state machine). The enrollment API, routing, app and
> wireless transfer come **after** that. The only part built today is the on-device
> enrollment core (`src/enrollment.py`) plus a dev wrapper (`scripts/enroll.py`).

## 1. Goal

Let the homeowner add a "friendly" person from the phone app. The Orin turns that
person's photos into a **template** (one 512-d vector) in the gallery, so the live
pipeline stops alarming on them. No model training or retraining is involved.

## 2. What exists now vs. later

| Piece | Status | Where |
|---|---|---|
| Photos → quality checks → template | **Built** | `src/enrollment.py` (`enroll_person`, `embed_photo`, `PhotoResult`) |
| Gallery storage + matching | **Built** | `src/gallery.py` (`make_template`, `save_gallery`, `load_gallery`, `match`) |
| Dev enrollment from folders | **Built** | `scripts/enroll.py` reads `data/enroll/<name>/` |
| Settings | **Built** | `config.yaml`: `min_face_px`, `enroll_min_photos`, `enroll_outlier_score` |
| Atomic gallery write + live reload | **Later phase**: only needed once enrollment can happen while the pipeline runs. For now `main.py` loads the gallery once at startup | `src/gallery.py`, `src/main.py` |
| Duplicate check (new person too close to an existing friend) | Planned | `src/enrollment.py` |
| Enrollment API service on the Orin | **Later phase** | n/a |
| Phone app: guided face scan + upload | **Later phase** (app side) | n/a |
| Pairing / authentication / transport security | **Later phase** | n/a |

## 3. End-to-end flow (later phase)

```
Phone app                                   Orin
─────────                                   ────
guided face scan (several poses)
  │  photos + name
  └──(home Wi-Fi, encrypted, paired device)──► enrollment service (later phase)
                                                 │  photos in memory
                                                 ▼
                                              src/enrollment.py   (same SCRFD + ArcFace as live)
                                                 │  template or rejection reasons
                                                 ▼
                                              src/gallery.py  → gallery.npz (atomic write)
                                                 │
                                                 ▼
                                              live pipeline reloads gallery
  ◄──────── result: enrolled / rejected + per-photo reasons ──┘
```

**Key rule:** the phone only **captures** photos; the **Orin computes every embedding**.
Embeddings from different models (or even different preprocessing) are not comparable,
so enrollment and live matching must use the exact same models on the same device.

## 4. Enrollment request state machine (later phase)

This is a **request lifecycle**, separate from the intrusion state machine
(`idle → verifying → FRIEND | INTRUDER`) in the live pipeline.

```
IDLE
 └─ request from app ─► AUTHENTICATING ── not owner / not paired ─────────► REJECTED
                          └─ ok ─► RECEIVING ── timeout / bad files ────────► REJECTED
                                     └─► PROCESSING   (embed_photo per photo)
                                           └─► QUALITY CHECK ── fails ─────► REJECTED (+ reasons)
                                                 └─ ok ─► COMMITTING
                                                            • atomic gallery write
                                                            • signal live pipeline to reload
                                                            └─► ENROLLED ─► IDLE
```

**Global guard:** refuse enrollment while the live pipeline is in an alarm (INTRUDER)
state. Otherwise an intruder with a stolen phone could enroll themselves to silence it.

Other operations the service will need: **add photos** to an existing friend (rebuild the
template), **remove friend** (delete their template; also a privacy requirement),
**list friends**.

## 5. Quality checks

| Check | Rejection reason (`PhotoResult.reason`) | Status |
|---|---|---|
| Image decodes | `could not read image` | Built |
| At least one face | `no face found` | Built |
| Exactly one face (can't tell who is enrolling otherwise) | `N faces, need exactly 1` | Built |
| Face width ≥ `min_face_px` | `face too small (X px < 40)` | Built |
| Photo agrees with the others (similarity to first template ≥ `enroll_outlier_score`) | `outlier: doesn't match the other photos` | Built |
| Enough good photos left (≥ `enroll_min_photos`) | person not enrolled (template is `None`) | Built |
| Not too similar to a *different* enrolled friend | n/a | Planned |
| Blur / exposure / pose variety | n/a | Optional, later |

**Known limitation:** outlier removal only works when most photos show the right person.
With 2 photos of two different people, each scores ~0.7 against their blended average and
passes. The check is meaningful from ~3 photos; a real product should require **~5+**.
`enroll_min_photos: 2` is a dev setting only.

## 6. Photo storage

| Device | Photos | Template vectors |
|---|---|---|
| **Our prototype** (dev laptop + our Orin) | Kept in `data/enroll/<name>/` (gitignored) so the gallery can be rebuilt if we switch face models | Stored in `gallery/gallery.npz` (gitignored) |
| **A production product** | Processed in memory, then discarded | Only these are stored (privacy, G4) |

Keeping photos is a **prototype convenience, not a production feature**. A model change in
production is not a realistic case for this project. `src/enrollment.py` never stores photos
itself; storing them is the caller's choice.

Templates are still biometric data: they identify a person. They stay on the device,
out of git, and are deleted when a friend is removed.

## 7. Updating the gallery while the pipeline runs (later phase)

- **Atomic write:** write `gallery.tmp.npz`, then rename over `gallery.npz`. A rename is
  all-or-nothing, so the live pipeline never reads a half-written file.
- **Reload:** the live pipeline re-reads the gallery when told to (or when the file's
  modification time changes), with no restart.

## 8. Security risks

| Risk | Mitigation (later phase) |
|---|---|
| Whoever can enroll a face can disable the alarm for that face | Owner-only, paired devices, encrypted transport, audit log of enroll/remove |
| Enrollment during an intrusion | Refuse while INTRUDER is active (section 4) |
| Photo of a friend fools the live camera (spoofing) | Out of scope for the POC; liveness detection is future work |
| Templates leaked from the device | Biometric data: keep local, restrict file access, delete on removal |

## 9. Open questions (for the API phase)

- Who builds and owns the enrollment service and app integration (group split)?
- Transport and pairing method (HTTP on the LAN, MQTT, BLE for pairing, ...)?
- Does the owner have to confirm each enrollment on the app before it is committed?
- How many photos and which poses does the app's guided scan capture?
- Where do the alarm state and the enrollment guard live if the system has several cameras?
