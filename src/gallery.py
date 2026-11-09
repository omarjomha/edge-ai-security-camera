"""Friend gallery: names -> template vectors. Build, save/load, match.

Vectors only (NumPy): no images, no models. Turning photos into embeddings is
src/face.py's job; the enrollment process lives in scripts/enroll.py.
"""
import os

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """For unit vectors, cosine similarity is just the dot product. -1..1, higher = same person."""
    return float(np.dot(a, b))


def make_template(embeddings: list[np.ndarray]) -> np.ndarray:
    """One person's embeddings -> one template (average direction, length 1)."""
    template = np.mean(embeddings, axis=0)
    return template / np.linalg.norm(template)  # mean of unit vectors is shorter than 1


def save_gallery(path: str, gallery: dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **gallery)              # one named array per friend


def load_gallery(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def match(vec: np.ndarray, gallery: dict[str, np.ndarray],
          threshold: float) -> tuple[str | None, float]:
    """Best-matching friend, or (None, score) if nobody reaches the threshold."""
    best_name, best_score = None, -1.0
    for name, template in gallery.items():
        score = cosine_similarity(vec, template)
        if score > best_score:
            best_name, best_score = name, score
    if best_score < threshold:
        return None, best_score
    return best_name, best_score
