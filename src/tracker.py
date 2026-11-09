"""Person tracking (ByteTrack-style) + per-track friend voting.

Simplification vs. real ByteTrack: motion is predicted with an alpha-beta filter
(velocity nudged by each prediction error) instead of a Kalman filter.
"""
from collections import Counter, deque

import numpy as np
from scipy.optimize import linear_sum_assignment

PERSON = 0                    # COCO class id
VELOCITY_GAIN = 0.3           # how strongly a prediction error corrects the velocity
LOST_VELOCITY_DECAY = 0.8     # unmatched tracks slow down instead of drifting away


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, 4) boxes vs (M, 4) boxes -> (N, M) IoU = overlap area / union area."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(tracks: list, boxes: np.ndarray, min_iou: float) -> tuple[list, list, list]:
    """Best one-to-one pairing (Hungarian algorithm) of tracks and boxes by IoU.

    Returns (pairs [(track_idx, box_idx)], unmatched track idxs, unmatched box idxs).
    """
    if not tracks or len(boxes) == 0:
        return [], list(range(len(tracks))), list(range(len(boxes)))
    ious = iou_matrix(np.array([t.box for t in tracks]), boxes)
    rows, cols = linear_sum_assignment(-ious)          # maximise total IoU
    pairs = [(r, c) for r, c in zip(rows, cols) if ious[r, c] >= min_iou]
    matched_t = {r for r, _ in pairs}
    matched_b = {c for _, c in pairs}
    return (pairs, [i for i in range(len(tracks)) if i not in matched_t],
            [j for j in range(len(boxes)) if j not in matched_b])


def assign_faces(tracks: list, face_boxes: np.ndarray) -> dict[int, int]:
    """Link each face to at most one track -> {track index: face index}.

    A face can belong to a track only if its centre is in the upper half of the
    person box; among valid tracks, the closest to the expected head position
    (top-centre of the box) wins. One face never counts for two people.
    """
    if not tracks or len(face_boxes) == 0:
        return {}
    INVALID = 1e6
    cost = np.full((len(tracks), len(face_boxes)), INVALID)
    for i, t in enumerate(tracks):
        x1, y1, x2, y2 = t.box
        w, h = x2 - x1, y2 - y1
        head = np.array([(x1 + x2) / 2, y1 + 0.15 * h])     # where a head usually is
        for j, (fx1, fy1, fx2, fy2, _) in enumerate(face_boxes):
            cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y1 + 0.5 * h:  # centre in the upper half
                cost[i, j] = np.hypot(cx - head[0], cy - head[1]) / w  # size-independent
    rows, cols = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] < INVALID}


class Track:
    def __init__(self, box: np.ndarray, score: float, vote_window: int):
        self.box = box.astype(np.float32)        # x1, y1, x2, y2 (frame pixels)
        self.velocity = np.zeros(4, np.float32)  # box change per frame
        self.score = score
        self.track_id = None                     # assigned once confirmed
        self.hits = 1                            # frames matched so far
        self.misses = 0                          # consecutive frames without a match
        # Recent face results: a friend's name, or None for "face seen, no match".
        # Frames without a usable face add nothing. Oldest votes drop out automatically.
        self.votes: deque[str | None] = deque(maxlen=vote_window)

    def add_vote(self, name: str | None) -> None:
        self.votes.append(name)

    def identity(self, min_matches: int) -> tuple[str | None, int]:
        """(friend name, votes for them) if a friend has >= min_matches recent votes,
        else (None, best friend's vote count)."""
        counts = Counter(v for v in self.votes if v is not None)
        if not counts:
            return None, 0
        name, count = counts.most_common(1)[0]
        return (name, count) if count >= min_matches else (None, count)

    def predict(self) -> None:
        self.box = self.box + self.velocity      # assume the person keeps moving the same way

    def update(self, box: np.ndarray, score: float) -> None:
        error = box - self.box                   # how far off the prediction was
        self.velocity += VELOCITY_GAIN * error
        self.box, self.score = box.astype(np.float32), score
        self.hits += 1
        self.misses = 0

    def mark_missed(self) -> None:
        self.misses += 1
        self.velocity *= LOST_VELOCITY_DECAY


class Tracker:
    def __init__(self, high_conf: float, new_conf: float, match_iou: float,
                 max_misses: int, min_hits: int, vote_window: int):
        self.high_conf = high_conf    # boxes >= this are "confident" (stage 1)
        self.new_conf = new_conf      # only boxes >= this may start a new track
        self.match_iou = match_iou    # minimum overlap to accept a stage-1 match
        self.max_misses = max_misses  # frames a lost track is kept before deletion
        self.min_hits = min_hits      # matched frames before a track gets an ID
        self.vote_window = vote_window  # face results remembered per track
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, dets: np.ndarray) -> list[Track]:
        """(N, 6) detections for one frame -> confirmed tracks seen in this frame."""
        people = dets[dets[:, 5] == PERSON]
        high = people[people[:, 4] >= self.high_conf]
        low = people[people[:, 4] < self.high_conf]

        for t in self.tracks:
            t.predict()

        # Stage 1: every track (including recently lost ones) vs. confident boxes
        pairs, left_t, left_h = match(self.tracks, high[:, :4], self.match_iou)
        for ti, bi in pairs:
            self.tracks[ti].update(high[bi, :4], high[bi, 4])

        # Stage 2 (the ByteTrack idea): tracks that were visible last frame but got no
        # confident box try the weak boxes (e.g. a half-hidden person). Strict IoU here.
        candidates = [self.tracks[i] for i in left_t if self.tracks[i].misses == 0]
        pairs2, _, _ = match(candidates, low[:, :4], min_iou=0.5)
        rescued = set()
        for ci, bi in pairs2:
            candidates[ci].update(low[bi, :4], low[bi, 4])
            rescued.add(id(candidates[ci]))

        for i in left_t:
            if id(self.tracks[i]) not in rescued:
                self.tracks[i].mark_missed()

        # New tracks only from unmatched, clearly confident boxes
        for bi in left_h:
            if high[bi, 4] >= self.new_conf:
                self.tracks.append(Track(high[bi, :4], high[bi, 4], self.vote_window))

        # Delete tracks lost for too long, and unconfirmed ones that already missed a
        # frame (likely a one-off false detection); give an ID to newly confirmed ones
        self.tracks = [t for t in self.tracks
                       if t.misses <= self.max_misses and not (t.track_id is None and t.misses > 0)]
        for t in self.tracks:
            if t.track_id is None and t.hits >= self.min_hits:
                t.track_id = self.next_id
                self.next_id += 1

        return [t for t in self.tracks if t.track_id is not None and t.misses == 0]
