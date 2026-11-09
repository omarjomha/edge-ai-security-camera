"""Enrollment: one person's photos -> quality-checked template vector.

Runs on the device (a future enrollment service calls it when the app uploads
photos), so it takes images already in memory, never a folder path. This module
never stores photos itself.

Photo storage:
- Our prototype (laptop + our Orin): photos are kept in data/enroll/<name>/
  (gitignored) so the gallery can be rebuilt if we switch face models.
  This is a prototype convenience, NOT a production feature.
- A production product would process photos in memory and discard them, storing
  only the template vectors in gallery.npz (privacy, G4).
"""
from dataclasses import dataclass

import numpy as np

from src.face import FaceDetector, FaceEmbedder, align_face
from src.gallery import cosine_similarity, make_template


@dataclass
class PhotoResult:
    """What happened to one enrollment photo (reported back to the user/app)."""
    photo_id: str                       # filename on dev, upload id from the app later
    accepted: bool
    reason: str                         # "ok" or why it was rejected
    embedding: np.ndarray | None = None
    score: float | None = None          # similarity to the final template


def embed_photo(face_det: FaceDetector, embedder: FaceEmbedder, photo_id: str,
                img: np.ndarray, min_face_px: int) -> PhotoResult:
    """Per-photo checks, then embed. Enrollment photos must show exactly one usable face."""
    if img is None:
        return PhotoResult(photo_id, False, "could not read image")
    boxes, landmarks = face_det.detect(img)
    if len(boxes) == 0:
        return PhotoResult(photo_id, False, "no face found")
    if len(boxes) > 1:                  # can't know which face is the person enrolling
        return PhotoResult(photo_id, False, f"{len(boxes)} faces, need exactly 1")
    face_px = int(boxes[0, 2] - boxes[0, 0])
    if face_px < min_face_px:
        return PhotoResult(photo_id, False, f"face too small ({face_px} px < {min_face_px})")
    vec = embedder.embed(align_face(img, landmarks[0]))
    return PhotoResult(photo_id, True, "ok", embedding=vec)


def enroll_person(face_det: FaceDetector, embedder: FaceEmbedder,
                  photos: list[tuple[str, np.ndarray]], min_face_px: int,
                  min_photos: int, outlier_score: float
                  ) -> tuple[np.ndarray | None, list[PhotoResult]]:
    """Photos of one person -> (template, per-photo results). Template is None if rejected."""
    results = [embed_photo(face_det, embedder, pid, img, min_face_px) for pid, img in photos]
    good = [r for r in results if r.accepted]
    if len(good) < min_photos:
        return None, results

    # Outlier removal: compare each photo to a first template, drop the ones that
    # disagree (wrong person, bad angle), then rebuild from the photos that remain
    first = make_template([r.embedding for r in good])
    for r in good:
        if cosine_similarity(r.embedding, first) < outlier_score:
            r.accepted, r.reason = False, "outlier: doesn't match the other photos"
    good = [r for r in good if r.accepted]
    if len(good) < min_photos:
        return None, results

    template = make_template([r.embedding for r in good])
    for r in results:
        if r.embedding is not None:
            r.score = cosine_similarity(r.embedding, template)
    return template, results
